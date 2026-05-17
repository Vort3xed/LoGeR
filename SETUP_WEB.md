# Real-Time Web Viewer — Setup & Usage

A two-page web app over WebSocket on top of the existing LoGeR `RealtimePipeline`. One tab captures camera frames (`/camera`); another renders the live 3D trajectory (`/viewer`).

```
[ browser /camera ]  ── JPEGs over WS ──▶  [ FastAPI :8000 ] ── poses+pcd over WS ──▶ [ browser /viewer ]
        ▲                                          │
        └───────────── extrap poses + anchor poses ┘  (the camera page also gets pose downlink for its 2D minimap)
```

## Prerequisites

1. The base LoGeR setup is complete — see `SETUP_REALTIME.md`. The conda env `loger` must have torch + the model checkpoints.
2. Node.js + npm (we install via conda for isolation):

```bash
$HOME/miniconda3/bin/conda install -n loger -c conda-forge nodejs=20 -y
```

3. Python deps already in `requirements.txt`: FastAPI, uvicorn, websockets, PIL.

## First run

Terminal A — backend (loads the model, ~30 s startup including torch.compile warmup):

```bash
conda activate loger
cd ~/Projects/LoGeR
python -m server.realtime_server
# → Uvicorn running on http://0.0.0.0:8000
```

Terminal B — frontend dev server:

```bash
conda activate loger
cd ~/Projects/LoGeR/web
npm install            # only the first time
npm run dev
# → Local: http://localhost:5173/
```

Open two browser tabs:
- `http://localhost:5173/camera` — requires camera permission. Streams JPEGs at 10 Hz.
- `http://localhost:5173/viewer` — 3D scene.

If you're on a remote machine, port-forward both:
```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 you@server
```

## Environment variables (backend)

| Var | Default | Purpose |
|---|---|---|
| `LOGER_CKPT` | `ckpts/LoGeR/latest.pt` | Model weights |
| `LOGER_CFG` | `ckpts/LoGeR/original_config.yaml` | Model config |
| `LOGER_WINDOW` | `8` | Streamer window size |
| `LOGER_DEVICE` | `cuda` | torch device |
| `LOGER_EMIT_PCD` | `1` | Set `0` to disable pointcloud broadcasts |
| `LOGER_PORT` | `8000` | uvicorn port |

## UI controls

### `/viewer`
- **Toggles** (top-right): true trajectory (cyan), interpolated trajectory (purple), point cloud.
- **Save run** — flushes the active recording to `runs/<name>_<id>.json.gz`. Includes anchors, extraps, and per-anchor pointclouds (quantized).
- **Reset trajectory** — clears the trajectory in the pipeline and recorder. Both viewer + camera tabs receive a `reset` event and clear their displays.
- **Saved runs list** (bottom-right) — click a run to overlay it on the viewer in place of the live stream. Hit *Resume live* to return.
- **Stats** (top-left): anchor Hz, extrap Hz, last inference ms, render Hz, lengths.

### `/camera`
- Streams every 100 ms (configurable via `SUBMIT_HZ` in `src/pages/Camera.tsx`).
- **Minimap** (top-right) — top-down (X-Z plane) live 2D trajectory. Cyan = anchors, purple = extraps, yellow dot = current position. Auto-zooms.
- **Stats** (top-left): uplink Hz, anchor Hz, extrap Hz, infer ms, lengths.

## Protocol summary

All downlink JSON. Uplink on `/ws/camera` is raw JPEG bytes per WS frame.

```json
// pose update — sent on /ws/viewer + /ws/camera
{ "type": "pose", "kind": "anchor" | "extrap",
  "idx": 42, "t": 1779008312.5,
  "pose": [16 floats c2w row-major],
  "fps": {"anchor": 3.1, "extrap": 9.8, "inference_ms": 310.4} }

// pointcloud — /ws/viewer only, after each anchor
{ "type": "pcd", "idx": 42, "n": 3000,
  "xyz": [9000 floats], "rgb": [9000 ints 0-255] | null }

// reset signal (both sockets)
{ "type": "reset" }
```

## Saved-run format

`runs/<name>_<id>.json.gz` (gzipped JSON):
```json
{
  "id": "abc12345", "name": "smoke",
  "started_at": 1779008282.7, "ended_at": 1779008312.2,
  "anchors": [{"idx", "t", "pose": [16 floats]}, ...],
  "extraps": [...],                        // sampled every ~5 frames to keep size small
  "pcds":    [{"idx", "xyz": {"q","scale","offset"}, "rgb": [...]}]
}
```

Pointclouds are quantized to int16 (~1 mm precision at ±32 m, ~10× smaller than float32). The server dequantizes on `GET /api/runs/:file` so the frontend gets float arrays.

## Production build

```bash
cd ~/Projects/LoGeR/web
npm run build       # → dist/
```

To serve the built UI from the backend instead of a separate Vite process, add a `StaticFiles` mount in `server/realtime_server.py`:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="web/dist", html=True), name="ui")
```
…and put it *after* all API + WS route declarations. Not done by default since `npm run dev` gives much faster iteration with HMR.

## Known limitations

- **WebRTC vs JPEG upload.** We send JPEGs over WS at 10 Hz. For an actual phone client at higher rates, WebRTC with a SFU would be more efficient. JPEG works fine on localhost for now.
- **getUserMedia requires HTTPS** in production. Use `localhost` for dev; for LAN testing, run behind an HTTPS-terminating proxy (caddy, ngrok).
- **First anchor takes 2–3 s after fresh connect** — the streamer buffer needs to fill (window=8 frames) before it commits an anchor pose. Until then, the viewer shows nothing and the minimap is empty.
- **VRAM.** Loading the model uses ~13 GB. If your GPU is contested, drop `LOGER_WINDOW=6` for ~10 GB usage.
