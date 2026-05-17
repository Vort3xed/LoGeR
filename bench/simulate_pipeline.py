"""End-to-end simulation of RealtimePipeline on the 50 office frames.

Feeds frames at a configurable "camera rate" (e.g. 30 FPS) while the model
runs in a background worker at its own rate (~3 FPS). For every submitted
frame, captures the pipeline's returned pose (predicted or precise) and
its latency. Reports:

- per-frame latency from submit to pose-available
- fraction of frames whose pose is extrapolated vs. inferred
- trajectory drift vs. batch model.forward reference
"""

import sys, glob, inspect, yaml, math, time, statistics, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.realtime_pipeline import RealtimePipeline, PipelineConfig
from loger.trajectory import align_poses_sim3, apply_sim3_to_pose


def load_model():
    with open("ckpts/LoGeR/original_config.yaml") as f:
        cfg = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in sig.parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    kwargs = {k: cfg[k] for k in cfg if k in valid}
    model = Pi3(**kwargs)
    state = torch.load("ckpts/LoGeR/latest.pt", map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to("cuda")


def build_frames(image_dir, n, pixel_limit=255_000):
    paths = natsorted(glob.glob(f"{image_dir}/*.png"))[:n]
    with Image.open(paths[0]) as im:
        W, H = im.size
    scale = math.sqrt(pixel_limit / (W * H))
    Wt, Ht = W * scale, H * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht: k -= 1
        else: m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    to_t = transforms.ToTensor()
    return [to_t(Image.open(p).convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS))
            for p in paths], (TW, TH)


def pose_diff(p_ref, p_test):
    td = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    Re = p_ref[:3, :3].T @ p_test[:3, :3]
    c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
    rd = math.degrees(math.acos(c))
    return td, rd


def main():
    print("[setup] loading model ...", flush=True)
    model = load_model()
    N = 50
    frames, _ = build_frames("data/examples/office", N)
    print(f"[data] {N} frames\n")

    # Reference: batch model.forward over all frames
    print("[ref] computing reference (batch model.forward W=8 OV=3) ...", flush=True)
    x = torch.stack(frames).unsqueeze(0).to("cuda")
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=8, overlap_size=3)
    ref_poses = out["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4)
    print(f"[ref] done\n")

    # Pipeline setup
    pipeline = RealtimePipeline(model, PipelineConfig(
        window_size=8, overlap_size=3, compile=True, inference_queue_size=1))

    # ---- WARM UP the streamer + compile BEFORE timing the "real-time" run ----
    print("[warmup] priming streamer with W+3 frames (triggers torch.compile)...", flush=True)
    t_warm = time.perf_counter()
    for fr in frames[:11]:
        _ = pipeline._streamer.update(fr.to("cuda"))
    print(f"[warmup] done in {time.perf_counter()-t_warm:.1f}s "
          f"(buffered {pipeline._streamer.n_buffered} frames)\n", flush=True)
    # Reset streamer state so the simulation starts clean
    pipeline._streamer.reset()
    pipeline._next_frame_idx = 0

    # Start worker
    pipeline.start()

    # Simulate camera at given FPS. With model ~300 ms/call, at 30 FPS we'd
    # process only ~1 frame per ~10 camera frames — most calls are extrapolation,
    # which is the realistic real-time scenario.
    CAMERA_FPS = 30
    FRAME_INTERVAL = 1.0 / CAMERA_FPS

    print(f"[sim] feeding {N} frames at {CAMERA_FPS} FPS "
          f"(model expected ~300 ms/call -> ~3 FPS inference)\n", flush=True)
    submit_latencies_ms = []
    extrapolated_flags = []
    t_start = time.perf_counter()
    for i, fr in enumerate(frames):
        target_t = t_start + i * FRAME_INTERVAL
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)
        timestamp = time.perf_counter() - t_start
        t_call_start = time.perf_counter()
        pose, was_extrapolated, anchor = pipeline.submit_frame(fr, timestamp)
        latency_ms = (time.perf_counter() - t_call_start) * 1000.0
        submit_latencies_ms.append(latency_ms)
        extrapolated_flags.append(was_extrapolated)

    print("[sim] draining inference queue ...", flush=True)
    time.sleep(1.0)
    pipeline.stop()
    t_total = time.perf_counter() - t_start
    print(f"\n[sim] complete in {t_total*1000:.0f} ms ({N/t_total:.2f} FPS submission rate)", flush=True)

    # Stats
    s = pipeline.stats()
    print("\n=== Pipeline stats ===")
    print(f"  frames submitted   : {s['submitted']}")
    print(f"  inferred frames    : {s['inferred']}    (model ran on these)")
    print(f"  dropped for inf    : {s['dropped']}    (newer frame arrived before worker free)")
    print(f"  extrapolated calls : {s['extrapolated']}  (foreground used prediction)")
    print(f"  trajectory length  : {s['trajectory_len']}")
    print(f"  last inference ms  : {s['last_inference_ms']:.1f}")

    # Latency: how fast is submit_frame()?
    print("\n=== submit_frame() latency (foreground) ===")
    lat = submit_latencies_ms
    print(f"  count: {len(lat)}")
    print(f"  mean:  {statistics.mean(lat):.2f} ms")
    print(f"  p50:   {statistics.median(lat):.2f} ms")
    print(f"  p95:   {sorted(lat)[int(0.95*len(lat))]:.2f} ms")
    print(f"  max:   {max(lat):.2f} ms")

    # Trajectory accuracy: align entire fused trajectory to reference, then per-frame drift
    fused = pipeline.trajectory.all_poses()
    if len(fused) >= 3:
        fused_idx = [i for i, _ in fused]
        fused_poses = np.stack([p for _, p in fused])
        ref_subset = np.stack([ref_poses[i] for i in fused_idx])
        scale, R, t = align_poses_sim3(fused_poses, ref_subset)
        aligned = np.stack([apply_sim3_to_pose(p, scale, R, t) for p in fused_poses])
        print(f"\n=== Trajectory accuracy (Sim(3)-aligned to reference) ===")
        rds, tds = [], []
        for i in range(len(fused_idx)):
            td, rd = pose_diff(ref_subset[i], aligned[i])
            rds.append(rd); tds.append(td)
        print(f"  rot RMSE: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.3f}°  "
              f"max: {max(rds):.3f}°")
        print(f"  trans RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.2f} mm  "
              f"max: {max(tds)*1000:.2f} mm")
        print(f"  scale alignment: {scale:.4f}")


if __name__ == "__main__":
    main()
