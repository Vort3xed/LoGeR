"""How much accuracy do we lose if we disable TTT and SWA entirely?

If the loss is small, we can build MaxSpeed without incremental TTT/SWA state
management (a major complexity reduction).

Compares (overlap=5, last-of-window pose):
  - bidir, TTT on, SWA on        (reference)
  - bidir, TTT off, SWA off      (turn_off_* flags)
  - causal, TTT off, SWA off     (the actual MaxSpeed target config)
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
sys.path.insert(0, str(REPO / "bench"))
from causal_attn_accuracy import install_causal_global_attention, load_model, build_frames


def pose_err(p_ref, p_test):
    td = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    Re = p_ref[:3, :3].T @ p_test[:3, :3]
    c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
    rd = math.degrees(math.acos(c))
    return td, rd


def run_window(model, win_tensor, win=8, ov=5, ttt_off=False, swa_off=False):
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(win_tensor, window_size=win, overlap_size=ov,
                    turn_off_ttt=ttt_off, turn_off_swa=swa_off)
    return out["camera_poses"][0, -1].float().cpu().numpy()


def main():
    print("[setup] loading models ...", flush=True)
    model_bi = load_model()
    model_ca = load_model()
    install_causal_global_attention(model_ca)

    WIN, OV = 8, 5
    N = 24
    frames, _ = build_frames("data/examples/office", N)
    print(f"[data] {N} frames\n")

    configs = [
        ("bidir, ttt+swa ON  (REF)", model_bi, False, False),
        ("bidir, ttt OFF",            model_bi, True,  False),
        ("bidir, swa OFF",            model_bi, False, True),
        ("bidir, ttt+swa OFF",        model_bi, True,  True),
        ("causal, ttt+swa OFF",       model_ca, True,  True),
        ("causal, ttt+swa ON",        model_ca, False, False),
    ]

    # Collect last-of-window pose for each config across sliding windows
    print(f"{'config':<30}  {'rot RMSE':>10}  {'rot max':>9}  {'t RMSE mm':>10}  {'t max mm':>9}")
    ref_poses = None
    for name, model, ttt_off, swa_off in configs:
        per_frame = []
        for T in range(WIN - 1, N):
            win = torch.stack(frames[T - WIN + 1:T + 1]).unsqueeze(0).to("cuda")
            p = run_window(model, win, WIN, OV, ttt_off, swa_off)
            per_frame.append(p)
        per_frame = np.stack(per_frame)
        if ref_poses is None:
            ref_poses = per_frame
            print(f"  {name:<30}  {'--':>10}  {'--':>9}  {'--':>10}  {'--':>9}")
            continue
        rds, tds = [], []
        for i in range(len(per_frame)):
            td, rd = pose_err(ref_poses[i], per_frame[i])
            rds.append(rd); tds.append(td)
        rot_rmse = math.sqrt(sum(r**2 for r in rds)/len(rds))
        t_rmse = math.sqrt(sum(t**2 for t in tds)/len(tds)) * 1000
        print(f"  {name:<30}  {rot_rmse:9.3f}°  {max(rds):8.3f}°  {t_rmse:9.2f}  {max(tds)*1000:8.2f}")


if __name__ == "__main__":
    main()
