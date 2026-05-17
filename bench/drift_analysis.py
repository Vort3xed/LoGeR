"""Long-run drift analysis on 50 office frames.

For each MaxSpeed candidate (smaller window, lower res), measure per-pose error
trend across the trajectory and decide whether errors compound, decay, or stay
flat. Conclusion drives whether MaxSpeed at that config is usable.

Reference = batch model.forward(all 50 frames, W=8, OV=3) — same as the demo.
For each frame index T in the global trajectory, we have:
  - reference_pose[T]: batch model's pose (stitched across windows internally)
  - candidate_pose[T]: streaming wrapper at smaller W (per-call last pose)

Both are camera-to-world. Compare directly (no alignment — we want absolute
drift, not after-the-fact alignment).
"""

import sys, glob, inspect, yaml, math, statistics, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.streaming import StreamingLoGeR, StreamingConfig


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


def streaming_trajectory(model, frames, W, OV):
    """Run the streaming wrapper across all frames. Return dict {frame_idx: 4x4 ndarray}."""
    streamer = StreamingLoGeR(model, StreamingConfig(
        window_size=W, overlap_size=OV, compile=False, persist_state=False))
    poses = {}
    for i, fr in enumerate(frames):
        p = streamer.update(fr.to("cuda"))
        if p is not None:
            poses[i] = p.numpy()
    return poses


def linear_trend(values):
    """Return slope of linear fit y = a + b*x. Positive slope = errors grow."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(values)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, values))
    den = sum((x-mx)**2 for x in xs)
    return num / den if den > 0 else 0.0


def analyze(name, ref_poses, test_poses):
    """Compare absolute pose errors and report drift trend."""
    common = sorted(set(ref_poses) & set(test_poses))
    rds, tds = [], []
    for T in common:
        td, rd = pose_diff(ref_poses[T], test_poses[T])
        tds.append(td); rds.append(rd)
    rot_slope = linear_trend(rds)
    t_slope = linear_trend(tds)
    print(f"\n--- {name} ({len(common)} frames compared) ---")
    print(f"  rot RMSE: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.3f}°  "
          f"max: {max(rds):.3f}°   mean: {sum(rds)/len(rds):.3f}°")
    print(f"  trans RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.2f} mm  "
          f"max: {max(tds)*1000:.2f} mm")
    print(f"  rot drift trend: {rot_slope:+.4f}° per frame   "
          f"(over {len(common)} frames → {rot_slope*len(common):+.2f}° total)")
    print(f"  t   drift trend: {t_slope*1000:+.3f} mm per frame   "
          f"(over {len(common)} frames → {t_slope*1000*len(common):+.1f} mm total)")
    # First half vs second half: are errors growing or shrinking?
    half = len(rds)//2
    early = sum(rds[:half])/half
    late = sum(rds[half:])/(len(rds)-half)
    delta = late - early
    print(f"  rot early-half mean: {early:.3f}°   late-half mean: {late:.3f}°   "
          f"Δ = {delta:+.3f}°")
    if abs(rot_slope * len(common)) < 1.0 and abs(delta) < 1.0:
        print(f"  -> ERRORS STABLE: no significant compounding observed")
    elif rot_slope > 0 and delta > 0:
        print(f"  -> ERRORS GROW: drift may compound over longer trajectories")
    else:
        print(f"  -> ERRORS DECREASE: stabilizing or improving over time")


def main():
    print("[setup] loading model ...", flush=True)
    model = load_model()
    frames, (TW, TH) = build_frames("data/examples/office", 50)
    print(f"[data] 50 frames @ {TW}x{TH}\n")

    # ===== Reference: batch model.forward over all 50 with W=8 OV=3 =====
    print("[ref] running batch model.forward(W=8, OV=3) over all 50 frames ...", flush=True)
    x = torch.stack(frames).unsqueeze(0).to("cuda")
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=8, overlap_size=3)
    ref_poses = {i: out["camera_poses"][0, i].float().cpu().numpy() for i in range(50)}
    print("[ref] done\n")

    # ===== Candidates =====
    candidates = [
        ("streaming W=8 OV=3 (MaxAccuracy)", 8, 3),
        ("streaming W=6 OV=3",                6, 3),
        ("streaming W=4 OV=1",                4, 1),
        ("streaming W=2 OV=0",                2, 0),
    ]
    for name, W, OV in candidates:
        test_poses = streaming_trajectory(model, frames, W, OV)
        analyze(name, ref_poses, test_poses)

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  - 'rot drift trend' is slope of per-frame error over the trajectory.")
    print("  - Near-zero slope means error is bounded; large positive slope means")
    print("    error grows and would be much worse over 20m of walking.")
    print("  - Compare early-half vs late-half means to check if errors compound.")


if __name__ == "__main__":
    main()
