"""More targeted test: under streaming semantics, measure the accuracy hit of
causal-mask approximation on the LATEST frame's pose — which is the only pose
the real-time server ever returns.

For each new frame T (starting at T=W-1), we run:
   bidir(frames[T-W+1 .. T])      -> pose for frame T
   causal(frames[T-W+1 .. T])     -> pose for frame T

and compare those two. This isolates the per-frame error a streaming server
would see in production.
"""

import sys, glob, inspect, yaml, math, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3

# import the install function from sibling script
sys.path.insert(0, str(REPO / "bench"))
from causal_attn_accuracy import install_causal_global_attention, load_model, build_frames


def main():
    print("[setup] loading models ...", flush=True)
    model_bi = load_model()
    model_ca = load_model()
    install_causal_global_attention(model_ca)

    WIN, OV = 8, 3
    N = 40
    frames, (TW, TH) = build_frames("data/examples/office", N)
    print(f"[data] {N} frames @ {TW}x{TH}\n")

    rot_diffs = []
    t_diffs = []
    print("[run] for each frame T>=W-1, comparing last-of-window pose ...", flush=True)
    print(f"{'T':>3}  {'t_bi':>20}  {'t-diff mm':>10}  {'rot deg':>9}")
    for T in range(WIN - 1, N):
        start = T - WIN + 1
        win = torch.stack(frames[start:T + 1]).unsqueeze(0).to("cuda")
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            out_bi = model_bi(win, window_size=WIN, overlap_size=OV)
            out_ca = model_ca(win, window_size=WIN, overlap_size=OV)
        p_bi = out_bi["camera_poses"][0, -1].float().cpu().numpy()
        p_ca = out_ca["camera_poses"][0, -1].float().cpu().numpy()
        td = float(np.linalg.norm(p_bi[:3, 3] - p_ca[:3, 3]))
        Re = p_bi[:3, :3].T @ p_ca[:3, :3]
        cos = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
        rd = math.degrees(math.acos(cos))
        rot_diffs.append(rd)
        t_diffs.append(td)
        if T < WIN + 4 or T >= N - 4 or T == N // 2:
            print(f"  {T:>3}  {str(np.round(p_bi[:3,3], 3)):>20}  {td*1000:9.2f}  {rd:8.3f}")

    print(f"\n=== Streaming last-of-window comparison over {len(rot_diffs)} frames ===")
    print(f"  trans  max: {max(t_diffs)*1000:.2f} mm   "
          f"mean: {sum(t_diffs)/len(t_diffs)*1000:.2f} mm   "
          f"RMSE: {math.sqrt(sum(t**2 for t in t_diffs)/len(t_diffs))*1000:.2f} mm")
    print(f"  rot    max: {max(rot_diffs):.3f} deg   "
          f"mean: {sum(rot_diffs)/len(rot_diffs):.3f} deg   "
          f"RMSE: {math.sqrt(sum(r**2 for r in rot_diffs)/len(rot_diffs)):.3f} deg")

    # Check for trend over time (does error grow as the trajectory progresses?)
    half = len(rot_diffs) // 2
    rd_early = sum(rot_diffs[:half]) / half
    rd_late = sum(rot_diffs[half:]) / (len(rot_diffs) - half)
    print(f"  rot mean early half: {rd_early:.3f} deg")
    print(f"  rot mean late  half: {rd_late:.3f} deg")
    if abs(rd_late - rd_early) < 0.5:
        print(f"  -> errors are stable over time (not accumulating)")
    elif rd_late > rd_early:
        print(f"  -> errors GROW over time by {(rd_late-rd_early):.2f} deg")
    else:
        print(f"  -> errors decrease over time")


if __name__ == "__main__":
    main()
