import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import Scene from "../components/Scene";
import Toggle from "../components/Toggle";
import { useTrajectoryStream, fetchRun } from "../lib/trajectory";
import type { Pcd } from "../lib/trajectory";
import type { SavedRunSummary } from "../lib/types";

export default function Viewer() {
  const stream = useTrajectoryStream("/ws/viewer", true);
  const [showAnchor, setShowAnchor] = useState(true);
  const [showExtrap, setShowExtrap] = useState(true);
  const [showPcd, setShowPcd] = useState(true);
  const [renderFps, setRenderFps] = useState(0);
  const [runs, setRuns] = useState<SavedRunSummary[]>([]);
  const [overlay, setOverlay] = useState<{
    anchors: Float32Array; anchorsLen: number;
    extraps: Float32Array; extrapsLen: number;
    pcds: Pcd[]; name: string;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/runs").then((r) => r.json()).then((d) => setRuns(d.runs || [])).catch(() => {});
  }, []);

  const onSave = async () => {
    setSaving(true);
    try {
      const r = await fetch("/api/save", { method: "POST" });
      const d = await r.json();
      setSaveMsg(`saved ${d.saved} (${d.anchors} anchors, ${d.pcds} pcds)`);
      const ru = await fetch("/api/runs").then((x) => x.json());
      setRuns(ru.runs || []);
    } catch (e) {
      setSaveMsg(`save failed: ${(e as Error).message}`);
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(null), 4000);
    }
  };

  const onReset = async () => {
    await fetch("/api/reset", { method: "POST" });
  };

  const onLoad = async (filename: string) => {
    try {
      const run = await fetchRun(filename);
      setOverlay({
        anchors: run.anchors, anchorsLen: run.anchorsLen,
        extraps: run.extraps, extrapsLen: run.extrapsLen,
        pcds: run.pcds, name: run.name,
      });
    } catch (e) {
      setSaveMsg(`load failed: ${(e as Error).message}`);
      setTimeout(() => setSaveMsg(null), 4000);
    }
  };

  // Choose which dataset to render: live stream or loaded overlay.
  const display = overlay ?? {
    anchors: stream.anchors, anchorsLen: stream.anchorsLen,
    extraps: stream.extraps, extrapsLen: stream.extrapsLen,
    pcds: stream.pcds, name: "(live)",
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <Scene
        anchors={display.anchors}
        anchorsLen={display.anchorsLen}
        extraps={display.extraps}
        extrapsLen={display.extrapsLen}
        pcds={display.pcds}
        showAnchor={showAnchor}
        showExtrap={showExtrap}
        showPcd={showPcd}
        resetTick={stream.resetTick + (overlay ? 1 : 0)}
        onRenderFps={setRenderFps}
      />

      {/* Top-left: nav + status + FPS */}
      <div className="absolute top-3 left-3 bg-black/60 backdrop-blur rounded-xl p-3 text-xs space-y-2 min-w-56">
        <div className="flex items-center justify-between gap-2">
          <Link to="/" className="text-sky-300 hover:text-sky-200">← home</Link>
          <span className={`px-2 py-0.5 rounded-full text-[10px] ${
            stream.status === "open" ? "bg-emerald-700" :
            stream.status === "connecting" ? "bg-amber-700" : "bg-rose-700"
          }`}>{stream.status}</span>
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
          <span className="text-neutral-400">anchor Hz</span><span>{stream.fps.anchor.toFixed(1)}</span>
          <span className="text-neutral-400">extrap Hz</span><span>{stream.fps.extrap.toFixed(1)}</span>
          <span className="text-neutral-400">infer ms</span><span>{stream.fps.inference_ms.toFixed(0)}</span>
          <span className="text-neutral-400">render Hz</span><span>{renderFps.toFixed(0)}</span>
          <span className="text-neutral-400">anchors</span><span>{display.anchorsLen}</span>
          <span className="text-neutral-400">extraps</span><span>{display.extrapsLen}</span>
          <span className="text-neutral-400">pcds</span><span>{display.pcds.length}</span>
        </div>
      </div>

      {/* Top-right: toggles + actions */}
      <div className="absolute top-3 right-3 bg-black/60 backdrop-blur rounded-xl p-3 text-sm space-y-3 min-w-56">
        <div className="space-y-1.5">
          <Toggle label="True trajectory (anchor)" on={showAnchor} onChange={setShowAnchor} swatch="#22d3ee" />
          <Toggle label="Interpolated (extrap)" on={showExtrap} onChange={setShowExtrap} swatch="#a78bfa" />
          <Toggle label="Point cloud" on={showPcd} onChange={setShowPcd} swatch="#f9fafb" />
        </div>
        <div className="border-t border-white/10 pt-3 space-y-2">
          <button
            onClick={onSave}
            disabled={saving}
            className="w-full px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50"
          >
            {saving ? "saving…" : "Save run"}
          </button>
          <button
            onClick={onReset}
            className="w-full px-3 py-1.5 rounded-md bg-rose-700 hover:bg-rose-600"
          >
            Reset trajectory
          </button>
          {overlay && (
            <button
              onClick={() => setOverlay(null)}
              className="w-full px-3 py-1.5 rounded-md bg-neutral-700 hover:bg-neutral-600"
            >
              Resume live ({overlay.name})
            </button>
          )}
          {saveMsg && <p className="text-xs text-emerald-300">{saveMsg}</p>}
        </div>
      </div>

      {/* Bottom-right: saved runs */}
      <div className="absolute bottom-3 right-3 bg-black/60 backdrop-blur rounded-xl p-3 text-xs space-y-1.5 max-h-72 overflow-y-auto min-w-64">
        <h3 className="text-sm font-semibold mb-1">Saved runs</h3>
        {runs.length === 0 && <p className="text-neutral-500">none yet</p>}
        {runs.map((r) => (
          <button
            key={r.file}
            onClick={() => onLoad(r.file)}
            className="w-full text-left px-2 py-1 rounded hover:bg-white/10 flex justify-between gap-2"
          >
            <span className="truncate">{r.name}</span>
            <span className="text-neutral-400 font-mono shrink-0">
              {r.anchors}a · {r.pcds}p
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
