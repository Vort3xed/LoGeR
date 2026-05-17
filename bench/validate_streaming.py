"""Validate StreamingLoGeR produces poses equivalent to a fresh single-window
model.forward() over the same frames.

We compare:
  - stream:  pose returned from StreamingLoGeR.update() for each new frame
  - ref:     model.forward(frames[k:k+W], window_size=W, overlap_size=O)[0, -1]
             (fresh forward with the latest W frames, take the last pose)

These should be IDENTICAL up to numerical noise when state persistence is off
(the only difference is the encoder is called frame-by-frame in streaming vs
all-frames-in-batch in the reference — bf16 matmul ordering can introduce
tiny float diffs).
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


def load_model(ckpt="ckpts/LoGeR/latest.pt", cfg="ckpts/LoGeR/original_config.yaml"):
    with open(cfg) as f:
        cfg_d = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in sig.parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    kwargs = {k: cfg_d[k] for k in cfg_d if k in valid}
    model = Pi3(**kwargs)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
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
    frames = []
    for p in paths:
        with Image.open(p) as im:
            frames.append(to_t(im.convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS)))
    return frames, (TW, TH)


def pose_diff(P_ref, P_test):
    """Return (translation_diff_norm, rotation_diff_deg) for two 4x4 poses."""
    t_diff = float(np.linalg.norm(P_ref[:3, 3] - P_test[:3, 3]))
    R_rel = P_ref[:3, :3].T @ P_test[:3, :3]
    cos = (np.trace(R_rel) - 1) / 2.0
    cos = max(-1.0, min(1.0, cos))
    rot_diff = math.degrees(math.acos(cos))
    return t_diff, rot_diff


def main():
    print("[setup] loading model ...", flush=True)
    model = load_model()

    N_TOTAL = 30
    WIN = 8
    OV = 3
    frames, (TW, TH) = build_frames("data/examples/office", N_TOTAL)
    print(f"[data] {N_TOTAL} frames @ {TW}x{TH}\n")

    # ===== STREAMING (compile ON, production config) =====
    print("[stream] feeding frames one by one (compile=True, production config)...",
          flush=True)
    streamer = StreamingLoGeR(model, StreamingConfig(
        window_size=WIN, overlap_size=OV, compile=True, persist_state=False))
    stream_poses = []  # list of (frame_idx, 4x4 ndarray)
    for i, fr in enumerate(frames):
        p = streamer.update(fr.to("cuda"))
        if p is not None:
            stream_poses.append((i, p.numpy()))
    print(f"[stream] got {len(stream_poses)} poses\n")

    # ===== REFERENCE: fresh single-window forward for each window =====
    print("[ref] computing reference poses via fresh model.forward per window ...",
          flush=True)
    ref_poses = []   # list of (frame_idx, 4x4 ndarray)
    for end_idx, _ in enumerate(stream_poses, start=WIN-1):
        # 'end_idx' is the index of the newest frame in the window
        start_idx = end_idx + 1 - WIN
        win_frames = torch.stack(frames[start_idx:end_idx + 1]).unsqueeze(0).to("cuda")
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(win_frames, window_size=WIN, overlap_size=OV)
        p = out["camera_poses"][0, -1].float().cpu().numpy()
        ref_poses.append((end_idx, p))
    print(f"[ref] got {len(ref_poses)} reference poses\n")

    # ===== compare =====
    print("=== Per-frame pose error (streaming vs fresh-forward reference) ===")
    print(f"{'frame':>6}  {'|t_ref|':>8}  {'|t-Δ| mm':>10}  {'rot Δ deg':>10}")
    t_diffs = []
    r_diffs = []
    for (i_s, p_s), (i_r, p_r) in zip(stream_poses, ref_poses):
        assert i_s == i_r
        t_norm = float(np.linalg.norm(p_r[:3, 3]))
        td, rd = pose_diff(p_r, p_s)
        t_diffs.append(td)
        r_diffs.append(rd)
        if i_s in (WIN - 1, (WIN - 1 + len(stream_poses)) // 2, stream_poses[-1][0]):
            print(f"  {i_s:>4}  {t_norm:>7.3f}  {td*1000:>9.4f}  {rd:>10.5f}")

    print(f"\n  max  t-diff: {max(t_diffs)*1000:.4f} mm   max rot-diff: {max(r_diffs):.5f} deg")
    print(f"  mean t-diff: {sum(t_diffs)/len(t_diffs)*1000:.4f} mm   "
          f"mean rot-diff: {sum(r_diffs)/len(r_diffs):.5f} deg")

    if max(t_diffs) < 1e-3 and max(r_diffs) < 0.1:
        print("\n  VERDICT: streaming output matches reference (mm + sub-tenth-deg level)")
    elif max(t_diffs) < 0.01 and max(r_diffs) < 1.0:
        print("\n  VERDICT: streaming output matches reference within float-precision noise")
    else:
        print("\n  VERDICT: streaming DIVERGES from reference. Investigate.")


if __name__ == "__main__":
    main()
