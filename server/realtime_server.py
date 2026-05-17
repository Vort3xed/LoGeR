"""FastAPI WebSocket server for the LoGeR real-time pipeline.

Endpoints
---------
GET  /healthz                        — liveness
GET  /api/runs                       — list saved runs
GET  /api/runs/{filename}            — load one saved run (JSON)
POST /api/save                       — flush the active recording to disk
POST /api/reset                      — drop trajectory + recording, fresh state
WS   /ws/camera                      — uplink JPEG frames; downlink pose updates
WS   /ws/viewer                      — downlink pose updates + pointclouds

Message protocol (downlink JSON, both sockets)
----------------------------------------------
  { "type": "pose",
    "kind": "anchor" | "extrap",
    "idx":  int,
    "t":    float (wall seconds),
    "pose": [16 floats c2w row-major],
    "fps":  { "anchor": float, "extrap": float, "inference_ms": float } }

  { "type": "pcd",                   — /ws/viewer only, after each anchor
    "idx":  int,
    "n":    int,
    "xyz":  [3n floats world meters],
    "rgb":  [3n ints 0-255] | null }

  { "type": "reset" }                — signal to clear displayed state

Uplink (binary on /ws/camera) is just raw JPEG bytes per WS message.
Optional control frames (text JSON):
  { "type": "config", "submit_hz": 10 }       — hint for the recorder
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as T

import inspect

import yaml

from loger.models.pi3 import Pi3
from loger.realtime_pipeline import RealtimePipeline, PipelineConfig
from server.persistence import RunRecorder, list_runs, load_run


def _load_pi3(ckpt_path: str, cfg_path: str, device: str) -> Pi3:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in sig.parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY)}
    kwargs = {k: cfg[k] for k in cfg if k in valid}
    model = Pi3(**kwargs)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)

log = logging.getLogger("loger.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# --------------------------------------------------------------------- state


class Hub:
    """Holds the pipeline + active broadcast subscribers + the recorder."""

    TARGET_H, TARGET_W = 434, 574        # full-resolution path

    def __init__(self):
        self.pipeline: Optional[RealtimePipeline] = None
        self.recorder = RunRecorder()
        self.recording = True
        self.viewer_clients: set[WebSocket] = set()
        self.camera_clients: set[WebSocket] = set()
        # Single asyncio loop used by the FastAPI event loop. Worker callbacks
        # (which run in a thread) marshal back to this loop with run_coroutine_threadsafe.
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Stats used for FPS readouts
        self._anchor_ts: list[float] = []
        self._extrap_ts: list[float] = []
        self._last_inference_ms: float = 0.0
        self._normalizer = T.Compose([
            T.ToTensor(),
            T.Resize((self.TARGET_H, self.TARGET_W), antialias=True),
        ])

    def init_pipeline(self, ckpt: str, cfg_path: str, device: str,
                       window_size: int, emit_pointcloud: bool) -> None:
        log.info("Loading model from %s (cfg=%s) …", ckpt, cfg_path)
        model = _load_pi3(ckpt, cfg_path, device)
        self.pipeline = RealtimePipeline(model, PipelineConfig(
            window_size=window_size, device=device, compile=True,
        ))
        self.pipeline.set_anchor_callback(
            self._on_anchor, emit_pointcloud=emit_pointcloud
        )

        # Warmup: run torch.compile tracing and fill the window buffer with
        # synthetic frames so the first real camera frame doesn't stall on a
        # 20-30 s compile pass while frames pile up and get dropped.
        log.info("Warming up streamer (compile + buffer fill, window=%d) …", window_size)
        dummy = torch.zeros(3, self.TARGET_H, self.TARGET_W, device=device)
        t0 = time.time()
        for i in range(window_size + 2):
            self.pipeline._streamer.update(dummy)
            if i == 0:
                log.info("  first forward done (compile trace): %.1f s",
                         time.time() - t0)
        # Reset trajectory so the synthetic poses aren't persisted.
        self.pipeline.trajectory.reset()
        self.pipeline._streamer.reset()
        self.recorder = RunRecorder()
        log.info("Warmup complete in %.1f s total.", time.time() - t0)

        self.pipeline.start()
        log.info("Pipeline ready (window=%d, device=%s, pcd=%s)",
                 window_size, device, emit_pointcloud)

    # ------------------------------------------------ ingest from /ws/camera

    def jpeg_to_tensor(self, jpeg_bytes: bytes) -> torch.Tensor:
        """JPEG -> (3, H, W) float in [0,1] at the model's target resolution.
        The streamer normalizes (mean/std) internally, so do NOT do it here."""
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        return self._normalizer(img)

    def submit_jpeg(self, jpeg_bytes: bytes, recv_time: float) -> tuple[np.ndarray, bool, int]:
        if self.pipeline is None:
            raise RuntimeError("pipeline not initialized")
        frame = self.jpeg_to_tensor(jpeg_bytes)
        return self.pipeline.submit_frame(frame, recv_time)

    # ------------------------------------------------ broadcast helpers

    def _now_fps(self, buf: list[float], window: float = 2.0) -> float:
        now = time.time()
        cutoff = now - window
        while buf and buf[0] < cutoff:
            buf.pop(0)
        return len(buf) / window if buf else 0.0

    def record_extrap(self, idx: int, t: float, pose: np.ndarray) -> None:
        self._extrap_ts.append(time.time())
        if self.recording:
            self.recorder.append_extrap(idx, t, pose)

    def fps_dict(self) -> dict:
        return {
            "anchor": round(self._now_fps(self._anchor_ts), 2),
            "extrap": round(self._now_fps(self._extrap_ts), 2),
            "inference_ms": round(self._last_inference_ms, 1),
        }

    def pose_message(self, kind: str, idx: int, t: float, pose: np.ndarray) -> dict:
        return {
            "type": "pose",
            "kind": kind,
            "idx": int(idx),
            "t": float(t),
            "pose": pose.flatten().astype(float).tolist(),
            "fps": self.fps_dict(),
        }

    # ----------------------------------------------- worker → asyncio bridge

    def _on_anchor(self, idx: int, pose: np.ndarray, pcd: Optional[dict]) -> None:
        if self.pipeline is not None:
            self._last_inference_ms = self.pipeline.stats()["last_inference_ms"]
        self._anchor_ts.append(time.time())
        if self.recording:
            self.recorder.append_anchor(idx, time.time(), pose)
            if pcd is not None and pcd.get("xyz") is not None:
                self.recorder.append_pcd(idx, pcd["xyz"], pcd.get("rgb"))

        pose_msg = self.pose_message("anchor", idx, time.time(), pose)
        pcd_msg = None
        if pcd is not None and pcd.get("xyz") is not None:
            xyz = pcd["xyz"]
            rgb = pcd.get("rgb")
            pcd_msg = {
                "type": "pcd",
                "idx": int(idx),
                "n": int(xyz.shape[0]),
                "xyz": xyz.flatten().tolist(),
                "rgb": rgb.flatten().tolist() if rgb is not None else None,
            }

        loop = self.loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_anchor(pose_msg, pcd_msg), loop
        )

    async def _broadcast_anchor(self, pose_msg: dict, pcd_msg: Optional[dict]) -> None:
        await self.broadcast_to(self.viewer_clients, pose_msg)
        await self.broadcast_to(self.camera_clients, pose_msg)
        if pcd_msg is not None:
            await self.broadcast_to(self.viewer_clients, pcd_msg)

    async def broadcast_to(self, subscribers: set[WebSocket], msg: dict) -> None:
        if not subscribers:
            return
        text = json.dumps(msg)
        dead: list[WebSocket] = []
        for ws in subscribers:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            subscribers.discard(ws)

    # ---------------------------------------------------- reset / lifecycle

    def reset(self) -> None:
        if self.pipeline is not None:
            self.pipeline.trajectory.reset()
        self.recorder = RunRecorder()
        self._anchor_ts.clear()
        self._extrap_ts.clear()


HUB = Hub()


# ----------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(_app: FastAPI):
    HUB.loop = asyncio.get_running_loop()
    import os
    ckpt = os.environ.get("LOGER_CKPT", "ckpts/LoGeR/latest.pt")
    cfg = os.environ.get("LOGER_CFG", "ckpts/LoGeR/original_config.yaml")
    device = os.environ.get("LOGER_DEVICE", "cuda")
    window = int(os.environ.get("LOGER_WINDOW", "8"))
    emit_pcd = os.environ.get("LOGER_EMIT_PCD", "1") != "0"
    HUB.init_pipeline(ckpt, cfg, device, window, emit_pcd)
    try:
        yield
    finally:
        if HUB.pipeline is not None:
            HUB.pipeline.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    if HUB.pipeline is None:
        return JSONResponse({"ok": False, "reason": "pipeline not loaded"}, status_code=503)
    return {"ok": True, "stats": HUB.pipeline.stats(), "fps": HUB.fps_dict()}


@app.get("/api/runs")
def api_runs():
    return {"runs": list_runs()}


@app.get("/api/runs/{filename}")
def api_get_run(filename: str):
    try:
        return load_run(filename)
    except FileNotFoundError:
        raise HTTPException(404, "not found")


@app.post("/api/save")
def api_save(name: Optional[str] = None):
    fname = HUB.recorder.flush_to_disk(name=name)
    return {"saved": fname, "anchors": len(HUB.recorder.anchors),
            "pcds": len(HUB.recorder.pcds)}


@app.post("/api/reset")
def api_reset():
    HUB.reset()
    asyncio.run_coroutine_threadsafe(
        HUB.broadcast_to(HUB.viewer_clients, {"type": "reset"}),
        HUB.loop,
    )
    asyncio.run_coroutine_threadsafe(
        HUB.broadcast_to(HUB.camera_clients, {"type": "reset"}),
        HUB.loop,
    )
    return {"ok": True}


# ----------------------------------------------------------------- ws


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    HUB.viewer_clients.add(ws)
    log.info("viewer connected (now %d)", len(HUB.viewer_clients))
    try:
        if HUB.pipeline is not None:
            latest = HUB.pipeline.trajectory.all_poses()
            for idx, pose in latest:
                await ws.send_text(json.dumps(HUB.pose_message("anchor", idx, time.time(), pose)))
        while True:
            try:
                _ = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("viewer ws error: %r", e)
    finally:
        HUB.viewer_clients.discard(ws)
        log.info("viewer disconnected (now %d)", len(HUB.viewer_clients))


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    HUB.camera_clients.add(ws)
    log.info("camera connected (now %d)", len(HUB.camera_clients))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
                t_recv = time.time()
                try:
                    pose, _is_extrap, _last_anchor = HUB.submit_jpeg(jpeg_bytes, t_recv)
                except Exception as e:
                    log.warning("submit_jpeg failed: %r", e)
                    continue
                if pose is not None:
                    HUB.record_extrap(HUB.pipeline.stats()["submitted"] - 1, t_recv, pose)
                    extrap_msg = HUB.pose_message(
                        "extrap", HUB.pipeline.stats()["submitted"] - 1, t_recv, pose,
                    )
                    # Send to camera (its minimap) and to all viewers.
                    await ws.send_text(json.dumps(extrap_msg))
                    await HUB.broadcast_to(HUB.viewer_clients, extrap_msg)
            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    continue
                if payload.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("camera ws error: %r", e)
    finally:
        HUB.camera_clients.discard(ws)
        log.info("camera disconnected (now %d)", len(HUB.camera_clients))


def main():
    import uvicorn
    host = "0.0.0.0"
    port = int((__import__("os").environ.get("LOGER_PORT", "8000")))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
