"""Real-time-feeling pipeline on top of LoGeR streaming + GlobalTrajectory.

Design
------
The model itself only updates poses at ~3 Hz (≈340 ms per call at W=8). If we
naively waited for inference on every camera frame, the application would feel
laggy by 300+ ms. To bridge that gap, we run the model in a background worker
thread and let the *foreground* return an extrapolated pose immediately based
on the most recent fused estimate.

  +-----------+      submit_frame()        +-----------------+
  | camera    | -------------------------> | RealtimePipeline |
  +-----------+                            |   (foreground)  |
                                           +---+-------------+
                                               |  extrapolated pose returned now
                                               |
                                               v
                                          +---------+
                                          | queue   | <- new frames arrive
                                          +----+----+
                                               |
                                       +-------v---------+
                                       | inference       |  <- runs StreamingLoGeR
                                       | worker thread   |     pulls frames, runs
                                       +-------+---------+     model.update(...)
                                               |
                                               v
                                       +-----------------+
                                       | GlobalTrajectory|  <- thread-safe Sim(3)
                                       | (fused state)   |     stitching
                                       +-----------------+

Concurrency notes
-----------------
- The inference worker is a Python thread. PyTorch GPU ops release the GIL,
  so the foreground keeps responsive even while the model is running.
- The GPU is shared with other agents in this environment. We do NOT pin
  memory or take other VRAM-greedy actions; the streaming wrapper's footprint
  is ~13 GB at peak, leaving ~30 GB for others.
- `submit_frame()` is non-blocking. If the inference queue is full (worker
  hasn't finished the previous frame yet), new frames are DROPPED for
  inference purposes, but the extrapolator continues to provide poses.

Output
------
Each `submit_frame()` returns:
    (predicted_pose, is_extrapolated, latest_anchor_idx)
where `predicted_pose` is the best estimate of the camera-to-world pose at
the moment of the frame arrival. `is_extrapolated=True` means inference is
still running and the pose is from the extrapolator; `False` means the
inference for this frame has completed and the pose is the precise one.

Most camera frames will return `is_extrapolated=True` because the camera
runs much faster than the model. Periodically (every model latency), one
returns the precise corrected pose.
"""

from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any

import numpy as np
import torch

from loger.streaming import StreamingLoGeR, StreamingConfig
from loger.trajectory import GlobalTrajectory


@dataclass
class PipelineConfig:
    window_size: int = 8
    overlap_size: int = 3
    autocast_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    compile: bool = True
    # Max frames buffered for inference. When full, new frames are dropped
    # for inference but still get an extrapolated pose. Set to 1 for tightest
    # latency — the worker always processes the freshest frame.
    inference_queue_size: int = 2


class RealtimePipeline:
    """Real-time pose tracking with a background inference worker.

    Usage::

        pipeline = RealtimePipeline(model, PipelineConfig())
        pipeline.start()
        for frame, timestamp in camera_stream:                # 3,H,W tensor + float seconds
            pose, was_extrapolated, last_anchor = pipeline.submit_frame(frame, timestamp)
            display(pose)
        pipeline.stop()
    """

    def __init__(self, model, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self._streamer = StreamingLoGeR(model, StreamingConfig(
            window_size=self.cfg.window_size,
            overlap_size=self.cfg.overlap_size,
            autocast_dtype=self.cfg.autocast_dtype,
            device=self.cfg.device,
            compile=self.cfg.compile,
            persist_state=False,
        ))
        self.trajectory = GlobalTrajectory(smoothing_alpha=1.0)

        self._inference_queue: "queue.Queue[Tuple[torch.Tensor, float, int]]" = \
            queue.Queue(maxsize=self.cfg.inference_queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()       # protects counters + state, not trajectory
        self._next_frame_idx = 0
        self._last_inferred_idx = -1
        self._last_inference_wall_ms = 0.0
        self._buf_cam_indices: list = []    # worker-only, no lock needed

        # Stats (all incremented under self._lock to avoid races between worker + main)
        self.dropped_frames = 0
        self.inferred_frames = 0
        self.extrapolated_frames = 0

        # Optional callback fired by worker when a new anchor is committed.
        # Signature: callback(anchor_frame_idx: int, anchor_pose: np.ndarray,
        #                     pointcloud: Optional[dict])
        # pointcloud dict has keys: "xyz" (K,3 float32), "rgb" (K,3 uint8 or None).
        # Called from worker thread — callback should be quick + thread-safe.
        self._anchor_callback: Optional[Callable[..., None]] = None
        self._emit_pointcloud: bool = False
        self._pcd_max_points: int = 3000

    def set_anchor_callback(self, cb, emit_pointcloud: bool = False,
                            max_points: int = 3000) -> None:
        """Register a fn to call when the worker commits a new anchor pose.

        If ``emit_pointcloud`` is True, the callback receives a third arg with
        a dict {"xyz": (K,3) np.float32, "rgb": (K,3) np.uint8 or None}.
        """
        self._anchor_callback = cb
        self._emit_pointcloud = emit_pointcloud
        self._pcd_max_points = max_points

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        assert self._worker is None, "RealtimePipeline already started"
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._inference_worker_loop,
            name="loger-inference-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._worker is None:
            return
        self._stop_event.set()
        # Push a sentinel to unblock queue.get
        try:
            self._inference_queue.put_nowait((None, 0.0, -1))  # type: ignore[arg-type]
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        self._worker = None

    # -------------------------------------------------------------- ingest

    def submit_frame(
        self,
        frame: torch.Tensor,
        timestamp: float,
    ) -> Tuple[Optional[np.ndarray], bool, int]:
        """Submit a new camera frame; return best-current pose.

        Returns:
            (pose 4x4 or None, is_extrapolated, last_anchor_idx)
        """
        with self._lock:
            frame_idx = self._next_frame_idx
            self._next_frame_idx += 1

        # Drop any older queued frame before adding the new one. We want the
        # worker to pick up the FRESHEST available frame, not the stalest.
        local_dropped = 0
        while True:
            try:
                self._inference_queue.get_nowait()
                local_dropped += 1
            except queue.Empty:
                break
        try:
            self._inference_queue.put_nowait((frame.detach(), timestamp, frame_idx))
        except queue.Full:
            local_dropped += 1  # rare race with worker
        with self._lock:
            self.dropped_frames += local_dropped

        # Foreground: predict from existing fused trajectory + recent velocity.
        latest = self.trajectory.latest()
        if latest is None:
            return None, True, -1
        last_anchor_idx, _ = latest
        predicted = self.trajectory.extrapolate(timestamp)
        if predicted is not None:
            with self._lock:
                self.extrapolated_frames += 1
            return predicted, True, last_anchor_idx
        # Fall back to last known pose if extrapolation failed
        return latest[1], True, last_anchor_idx

    # ------------------------------------------------------------ worker

    def _inference_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, timestamp, frame_idx = self._inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break  # sentinel

            # Drain queue → always work on the FRESHEST available frame
            local_dropped = 0
            while True:
                try:
                    frame2, ts2, idx2 = self._inference_queue.get_nowait()
                    if frame2 is None:
                        self._stop_event.set()
                        break
                    local_dropped += 1
                    frame, timestamp, frame_idx = frame2, ts2, idx2
                except queue.Empty:
                    break

            t0 = time.perf_counter()
            try:
                _ = self._streamer.update(frame.to(self.cfg.device, non_blocking=True))
            except Exception:
                raise
            inf_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._last_inference_wall_ms = inf_ms
                self.inferred_frames += 1
                self.dropped_frames += local_dropped

            self._buf_cam_indices.append(frame_idx)
            while len(self._buf_cam_indices) > self.cfg.window_size:
                self._buf_cam_indices.pop(0)

            # Stitch immediately. (Previous version tried to defer this for
            # CPU/GPU overlap, but streamer.update() already synchronizes via
            # .detach().cpu(), so there's nothing to overlap.)
            if self._streamer.is_ready:
                win = self._streamer.latest_window_poses()
                if win is not None:
                    # Per-frame model confidence (W,) — used to downweight
                    # low-confidence anchors during Sim(3) stitching. Returns
                    # None if the model lacks a conf head, in which case
                    # ingest_window_sparse falls back to unweighted alignment.
                    conf = self._streamer.latest_window_confidence()
                    conf_np = conf.numpy() if conf is not None else None
                    self.trajectory.ingest_window_sparse(
                        win.numpy(),
                        cam_indices=list(self._buf_cam_indices),
                        timestamp=timestamp,
                        confidence=conf_np,
                    )
                    with self._lock:
                        self._last_inferred_idx = frame_idx
                    # Notify the world: a new anchor pose just landed.
                    if self._anchor_callback is not None:
                        try:
                            anchor_pose = self.trajectory.get_pose(frame_idx)
                            if anchor_pose is not None:
                                pcd = None
                                if self._emit_pointcloud:
                                    res = self._streamer.latest_newest_pointcloud(
                                        max_points=self._pcd_max_points
                                    )
                                    if res is not None:
                                        xyz, rgb = res
                                        pcd = {
                                            "xyz": xyz.numpy().astype(np.float32),
                                            "rgb": rgb.numpy() if rgb is not None else None,
                                        }
                                self._anchor_callback(frame_idx, anchor_pose, pcd)
                        except Exception:
                            import traceback
                            traceback.print_exc()
                            # callback errors must not crash the worker

    # --------------------------------------------------------------- stats

    def stats(self) -> dict:
        with self._lock:
            return {
                "submitted": self._next_frame_idx,
                "inferred": self.inferred_frames,
                "extrapolated": self.extrapolated_frames,
                "dropped": self.dropped_frames,
                "last_inference_ms": self._last_inference_wall_ms,
                "last_inferred_idx": self._last_inferred_idx,
                "trajectory_len": len(self.trajectory),
            }
