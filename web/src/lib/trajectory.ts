// Shared rolling trajectory state.
//
// We keep two parallel time series:
//   - anchors[]: model-derived ground-truth-from-server poses (3 Hz-ish)
//   - extraps[]: extrapolated foreground poses (camera-rate)
//
// Both are plain arrays of [x, y, z] for cheap line rendering. Three.js
// re-uses these arrays directly via BufferGeometry.setAttribute.

import { useEffect, useMemo, useRef, useState } from "react";

import { openWS } from "./ws";
import type { DownlinkMsg, PoseMsg, PcdMsg, FpsBlock } from "./types";

const MAX_POINTS = 8000;          // soft cap per line — older points get dropped
const MAX_PCDS = 60;              // anchor pointclouds kept in memory
const MAX_PCD_POINTS = 3000;      // per-anchor (matches server max)

export type Pcd = {
  idx: number;
  xyz: Float32Array;
  rgb: Uint8Array | null;
};

export type TrajectoryState = {
  anchors: Float32Array;            // 3*N
  anchorsLen: number;
  extraps: Float32Array;            // 3*M
  extrapsLen: number;
  pcds: Pcd[];
  fps: FpsBlock;
  status: "connecting" | "open" | "closed";
  resetTick: number;                // bumps when reset arrives
};

function unpackPosition(pose: number[]): [number, number, number] {
  // Row-major 4x4 c2w: translation is at indices 3, 7, 11.
  return [pose[3], pose[7], pose[11]];
}

export function useTrajectoryStream(path = "/ws/viewer", subscribePcd = true) {
  const anchors = useRef(new Float32Array(MAX_POINTS * 3));
  const extraps = useRef(new Float32Array(MAX_POINTS * 3));
  const anchorsLen = useRef(0);
  const extrapsLen = useRef(0);
  const pcdsRef = useRef<Pcd[]>([]);
  const seenAnchorIdx = useRef<Set<number>>(new Set());

  const [tick, setTick] = useState(0);
  const [resetTick, setResetTick] = useState(0);
  const [fps, setFps] = useState<FpsBlock>({ anchor: 0, extrap: 0, inference_ms: 0 });
  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");

  useEffect(() => {
    const pushAnchor = (m: PoseMsg) => {
      if (seenAnchorIdx.current.has(m.idx)) return;
      seenAnchorIdx.current.add(m.idx);
      const [x, y, z] = unpackPosition(m.pose);
      if (anchorsLen.current >= MAX_POINTS) {
        anchors.current.copyWithin(0, 3, anchorsLen.current * 3);
        anchorsLen.current -= 1;
      }
      const i = anchorsLen.current * 3;
      anchors.current[i] = x; anchors.current[i + 1] = y; anchors.current[i + 2] = z;
      anchorsLen.current += 1;
    };
    const pushExtrap = (m: PoseMsg) => {
      const [x, y, z] = unpackPosition(m.pose);
      if (extrapsLen.current >= MAX_POINTS) {
        extraps.current.copyWithin(0, 3, extrapsLen.current * 3);
        extrapsLen.current -= 1;
      }
      const i = extrapsLen.current * 3;
      extraps.current[i] = x; extraps.current[i + 1] = y; extraps.current[i + 2] = z;
      extrapsLen.current += 1;
    };
    const pushPcd = (m: PcdMsg) => {
      if (!subscribePcd) return;
      const n = Math.min(m.n, MAX_PCD_POINTS);
      const xyz = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) xyz[i] = m.xyz[i];
      const rgb = m.rgb ? new Uint8Array(m.rgb.slice(0, n * 3)) : null;
      pcdsRef.current.push({ idx: m.idx, xyz, rgb });
      while (pcdsRef.current.length > MAX_PCDS) pcdsRef.current.shift();
    };

    const ws = openWS(path, (msg: DownlinkMsg) => {
      switch (msg.type) {
        case "pose":
          if (msg.kind === "anchor") pushAnchor(msg);
          else pushExtrap(msg);
          setFps(msg.fps);
          setTick((t) => (t + 1) % 1_000_000);
          break;
        case "pcd":
          pushPcd(msg);
          setTick((t) => (t + 1) % 1_000_000);
          break;
        case "reset":
          anchorsLen.current = 0;
          extrapsLen.current = 0;
          pcdsRef.current = [];
          seenAnchorIdx.current = new Set();
          setResetTick((t) => t + 1);
          setTick((t) => (t + 1) % 1_000_000);
          break;
        default:
          break;
      }
    }, setStatus);

    return () => ws.close();
  }, [path, subscribePcd]);

  return useMemo<TrajectoryState>(() => ({
    anchors: anchors.current,
    anchorsLen: anchorsLen.current,
    extraps: extraps.current,
    extrapsLen: extrapsLen.current,
    pcds: pcdsRef.current,
    fps,
    status,
    resetTick,
  }), [tick, fps, status, resetTick]);
}

export type LoadedRun = {
  id: string;
  name: string;
  anchors: Float32Array;
  anchorsLen: number;
  extraps: Float32Array;
  extrapsLen: number;
  pcds: Pcd[];
};

export async function fetchRun(filename: string): Promise<LoadedRun> {
  const res = await fetch(`/api/runs/${encodeURIComponent(filename)}`);
  if (!res.ok) throw new Error(`load failed: ${res.status}`);
  const data = await res.json();
  const anchors = new Float32Array(data.anchors.length * 3);
  for (let i = 0; i < data.anchors.length; i++) {
    const p = data.anchors[i].pose;
    anchors[i * 3] = p[3]; anchors[i * 3 + 1] = p[7]; anchors[i * 3 + 2] = p[11];
  }
  const extraps = new Float32Array(data.extraps.length * 3);
  for (let i = 0; i < data.extraps.length; i++) {
    const p = data.extraps[i].pose;
    extraps[i * 3] = p[3]; extraps[i * 3 + 1] = p[7]; extraps[i * 3 + 2] = p[11];
  }
  const pcds: Pcd[] = (data.pcds || []).map((p: any) => ({
    idx: p.idx,
    xyz: new Float32Array(p.xyz),
    rgb: p.rgb ? new Uint8Array(p.rgb) : null,
  }));
  return {
    id: data.id, name: data.name,
    anchors, anchorsLen: data.anchors.length,
    extraps, extrapsLen: data.extraps.length,
    pcds,
  };
}
