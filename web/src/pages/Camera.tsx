// /camera page: capture from device camera, stream JPEGs over WS, show preview
// + 2D minimap of the trajectory feedback the server sends back.
//
// Single WS connection serves both directions: outgoing JPEG binary frames
// and incoming JSON pose messages. The minimap reads from the same
// trajectory buffers as the 3D viewer would.

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import Minimap from "../components/Minimap";
import type { DownlinkMsg, FpsBlock, PoseMsg } from "../lib/types";

const SUBMIT_HZ = 10;
const QUALITY = 0.7;
const MAX_POINTS = 8000;

function unpackXYZ(pose: number[]): [number, number, number] {
  return [pose[3], pose[7], pose[11]];
}

export default function Camera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Buffers shared with the minimap (mutated in place; minimap re-reads every frame).
  const anchorsBuf = useRef(new Float32Array(MAX_POINTS * 3));
  const extrapsBuf = useRef(new Float32Array(MAX_POINTS * 3));
  const anchorsLenRef = useRef(0);
  const extrapsLenRef = useRef(0);
  const seenAnchorIdx = useRef<Set<number>>(new Set());

  const [tick, setTick] = useState(0);                  // forces minimap rerender
  const [resetTick, setResetTick] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uplinkHz, setUplinkHz] = useState(0);
  const [fps, setFps] = useState<FpsBlock>({ anchor: 0, extrap: 0, inference_ms: 0 });

  // ---- Camera capture -------------------------------------------------

  useEffect(() => {
    let mediaStream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) { mediaStream.getTracks().forEach((t) => t.stop()); return; }
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          await videoRef.current.play();
        }
      } catch (e: any) {
        setError(`camera error: ${e?.message || e}`);
      }
    })();
    return () => {
      cancelled = true;
      mediaStream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ---- Single WebSocket: uplink JPEGs + downlink poses ---------------

  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/camera`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    const pushAnchor = (m: PoseMsg) => {
      if (seenAnchorIdx.current.has(m.idx)) return;
      seenAnchorIdx.current.add(m.idx);
      const [x, y, z] = unpackXYZ(m.pose);
      if (anchorsLenRef.current >= MAX_POINTS) {
        anchorsBuf.current.copyWithin(0, 3, anchorsLenRef.current * 3);
        anchorsLenRef.current -= 1;
      }
      const i = anchorsLenRef.current * 3;
      anchorsBuf.current[i] = x; anchorsBuf.current[i + 1] = y; anchorsBuf.current[i + 2] = z;
      anchorsLenRef.current += 1;
    };
    const pushExtrap = (m: PoseMsg) => {
      const [x, y, z] = unpackXYZ(m.pose);
      if (extrapsLenRef.current >= MAX_POINTS) {
        extrapsBuf.current.copyWithin(0, 3, extrapsLenRef.current * 3);
        extrapsLenRef.current -= 1;
      }
      const i = extrapsLenRef.current * 3;
      extrapsBuf.current[i] = x; extrapsBuf.current[i + 1] = y; extrapsBuf.current[i + 2] = z;
      extrapsLenRef.current += 1;
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg: DownlinkMsg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === "pose") {
        if (msg.kind === "anchor") pushAnchor(msg);
        else pushExtrap(msg);
        setFps(msg.fps);
        setTick((t) => (t + 1) % 1_000_000);
      } else if (msg.type === "reset") {
        anchorsLenRef.current = 0;
        extrapsLenRef.current = 0;
        seenAnchorIdx.current = new Set();
        setResetTick((t) => t + 1);
        setTick((t) => (t + 1) % 1_000_000);
      }
    };

    let interval: number | null = null;
    let sentLastSec = 0;
    let lastReport = performance.now();

    ws.onopen = () => {
      setStreaming(true);
      interval = window.setInterval(async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2 || ws.readyState !== WebSocket.OPEN) return;
        const w = video.videoWidth, h = video.videoHeight;
        if (!w || !h) return;
        const targetW = 640;
        const ar = w / h;
        canvas.width = targetW;
        canvas.height = Math.round(targetW / ar);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob: Blob | null = await new Promise((r) =>
          canvas.toBlob(r, "image/jpeg", QUALITY),
        );
        if (!blob) return;
        if (ws.readyState !== WebSocket.OPEN) return;
        const ab = await blob.arrayBuffer();
        ws.send(ab);
        sentLastSec += 1;
        const now = performance.now();
        if (now - lastReport > 1000) {
          setUplinkHz(sentLastSec * 1000 / (now - lastReport));
          sentLastSec = 0;
          lastReport = now;
        }
      }, Math.round(1000 / SUBMIT_HZ));
    };
    ws.onclose = () => {
      setStreaming(false);
      if (interval !== null) clearInterval(interval);
    };
    ws.onerror = () => setError("websocket error — is the server running?");

    return () => {
      if (interval !== null) clearInterval(interval);
      try { ws.close(); } catch { /* noop */ }
    };
  }, []);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />
      <canvas ref={canvasRef} className="hidden" />

      {/* Top-left HUD */}
      <div className="absolute top-3 left-3 bg-black/60 backdrop-blur rounded-xl p-3 text-xs space-y-1.5 max-w-xs">
        <div className="flex items-center justify-between gap-2">
          <Link to="/" className="text-sky-300 hover:text-sky-200">← home</Link>
          <span className={`px-2 py-0.5 rounded-full text-[10px] ${
            streaming ? "bg-emerald-700" : "bg-rose-700"
          }`}>{streaming ? "streaming" : "offline"}</span>
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
          <span className="text-neutral-400">uplink Hz</span><span>{uplinkHz.toFixed(1)}</span>
          <span className="text-neutral-400">anchor Hz</span><span>{fps.anchor.toFixed(1)}</span>
          <span className="text-neutral-400">extrap Hz</span><span>{fps.extrap.toFixed(1)}</span>
          <span className="text-neutral-400">infer ms</span><span>{fps.inference_ms.toFixed(0)}</span>
          <span className="text-neutral-400">anchors</span><span>{anchorsLenRef.current}</span>
          <span className="text-neutral-400">extraps</span><span>{extrapsLenRef.current}</span>
        </div>
        {error && <p className="text-rose-300">{error}</p>}
        <p className="text-neutral-400 text-[10px] pt-1">
          tip: open /viewer on another tab for the 3D view.
        </p>
      </div>

      {/* Top-right minimap */}
      <div className="absolute top-3 right-3">
        <Minimap
          anchors={anchorsBuf.current}
          anchorsLen={anchorsLenRef.current}
          extraps={extrapsBuf.current}
          extrapsLen={extrapsLenRef.current}
          resetTick={resetTick}
        />
      </div>

      {/* Tick is intentionally unused; it just exists to drive rerenders when buffers mutate. */}
      <div className="hidden">{tick}</div>
    </div>
  );
}
