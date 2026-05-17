"""Validate GlobalTrajectory stitching produces a globally consistent trajectory.

For each new frame T, the streaming wrapper outputs a (W, 4, 4) window of
poses in the local frame of frame T-W+1. We feed this window to
GlobalTrajectory.ingest_window(), which stitches it into a single global
frame using Sim(3) alignment on the W-1 overlapping frames.

We then compare the stitched global trajectory to the batch
model.forward(all_frames, W=8, OV=3) reference (which uses the model's
own internal Sim(3) merging).
"""

import sys, glob, inspect, yaml, math, time, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.streaming import StreamingLoGeR, StreamingConfig
from loger.trajectory import GlobalTrajectory


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

    WIN, OV = 8, 3

    # ===== Reference: batch model.forward(all 50 frames) =====
    print("[ref] batch forward ...", flush=True)
    x = torch.stack(frames).unsqueeze(0).to("cuda")
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=WIN, overlap_size=OV)
    ref_poses = {i: out["camera_poses"][0, i].float().cpu().numpy() for i in range(N)}
    print("[ref] done\n")

    # ===== Streaming + stitching =====
    print("[stitch] streaming + GlobalTrajectory ...", flush=True)
    streamer = StreamingLoGeR(model, StreamingConfig(
        window_size=WIN, overlap_size=OV, compile=False, persist_state=False))
    traj = GlobalTrajectory(smoothing_alpha=1.0)  # overwrite each time for now

    for i, fr in enumerate(frames):
        _ = streamer.update(fr.to("cuda"))
        if not streamer.is_ready:
            continue
        # Pull the latest window's poses; ingest as a window into the trajectory
        win = streamer.latest_window_poses()  # (W, 4, 4)
        if win is None:
            continue
        traj.ingest_window(win.numpy(), newest_idx=i, timestamp=float(i))
    print(f"[stitch] traj has {len(traj)} frames\n")

    # ===== Compare =====
    # Align the entire global trajectory to the batch reference once (Sim3 on translations)
    # — this absorbs the global frame's arbitrary origin/orientation, leaving only
    # the actual relative-pose accuracy in the residual.
    all_poses = traj.all_poses()  # list of (idx, 4x4)
    print(f"compared poses (after stitch):")
    print(f"{'idx':>4}  {'rot diff deg':>14}  {'t diff mm':>12}")
    rds, tds = [], []
    for idx, p in all_poses:
        td, rd = pose_diff(ref_poses[idx], p)
        rds.append(rd); tds.append(td)
        if idx % 10 == 0 or idx in (WIN - 1, N - 1):
            print(f"  {idx:>3}  {rd:13.4f}  {td*1000:11.3f}")
    print(f"\n  rot RMSE: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.4f}°    "
          f"max: {max(rds):.4f}°    mean: {sum(rds)/len(rds):.4f}°")
    print(f"  t RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.3f} mm    "
          f"max: {max(tds)*1000:.3f} mm")

    # Drift trend
    n = len(rds)
    half = n // 2
    print(f"  early-half rot mean: {sum(rds[:half])/half:.4f}°    "
          f"late-half rot mean: {sum(rds[half:])/(n-half):.4f}°    "
          f"Δ = {sum(rds[half:])/(n-half) - sum(rds[:half])/half:+.4f}°")


if __name__ == "__main__":
    main()
