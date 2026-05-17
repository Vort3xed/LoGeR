"""Quantify how much accuracy we lose at lower resolution.

We treat the FULL-resolution (default 255k pixel limit) output as ground truth
and measure how the lower-resolution outputs compare:
  - translation RMSE (m, after Umeyama Sim(3) alignment via evo)
  - rotation RMSE (deg)
  - per-frame translation max error
"""

import sys, glob, inspect, yaml, torch, math
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3


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


def build_input(image_dir, n, pixel_limit):
    paths = natsorted(glob.glob(f"{image_dir}/*.png"))[:n]
    with Image.open(paths[0]) as im:
        W, H = im.size
    scale = math.sqrt(pixel_limit / (W * H))
    Wt, Ht = W * scale, H * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    out = torch.empty((len(paths), 3, TH, TW), dtype=torch.float32)
    to_t = transforms.ToTensor()
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            out[i].copy_(to_t(img.convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS)))
    return out.unsqueeze(0).to("cuda"), (TW, TH)


def forward_poses(model, x, win=8, ov=3):
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=win, overlap_size=ov)
    # camera_poses shape: (B, N, 4, 4) — camera-to-world
    return out["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4)


def umeyama_align(src, dst, with_scale=True):
    """Estimate similarity transform mapping src -> dst.
    Both (N,3). Returns (s, R, t) such that s*R@src.T + t.T ≈ dst.T"""
    mu_s = src.mean(0); mu_d = dst.mean(0)
    s = src - mu_s; d = dst - mu_d
    H = s.T @ d / len(src)
    U, S, Vt = np.linalg.svd(H)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = Vt.T @ Sgn @ U.T
    if with_scale:
        var_s = (s ** 2).sum() / len(src)
        scale = (S * np.diag(Sgn)).sum() / var_s
    else:
        scale = 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def rotation_error_deg(R_ref, R_test):
    """Per-pose rotation error in degrees."""
    n = len(R_ref)
    errs = []
    for i in range(n):
        # relative rotation: R_err = R_ref^T @ R_test
        R_err = R_ref[i].T @ R_test[i]
        cos = (np.trace(R_err) - 1) / 2.0
        cos = max(-1.0, min(1.0, cos))
        errs.append(math.degrees(math.acos(cos)))
    return np.array(errs)


def evaluate(poses_ref, poses_test):
    t_ref = poses_ref[:, :3, 3]      # (N, 3)
    t_test = poses_test[:, :3, 3]
    R_ref = poses_ref[:, :3, :3]
    R_test = poses_test[:, :3, :3]

    # Sim(3) align test->ref by translations
    scale, R, t = umeyama_align(t_test, t_ref, with_scale=True)
    t_test_aligned = (scale * (R @ t_test.T)).T + t

    trans_err = np.linalg.norm(t_test_aligned - t_ref, axis=1)
    ate = math.sqrt((trans_err ** 2).mean())

    # Align rotations: R_aligned = R @ R_test (since translations had R applied)
    R_test_aligned = np.array([R @ Rt for Rt in R_test])
    rot_errs = rotation_error_deg(R_ref, R_test_aligned)

    return {
        "scale_align": scale,
        "ate_m": ate,
        "trans_max_m": float(trans_err.max()),
        "trans_mean_m": float(trans_err.mean()),
        "rot_rmse_deg": float(math.sqrt((rot_errs ** 2).mean())),
        "rot_max_deg": float(rot_errs.max()),
    }


def main():
    print("[setup] loading model ...", flush=True)
    model = load_model()
    DATA = "data/examples/office"
    N = 8
    WIN, OV = 8, 3

    # Reference: full-resolution (default demo behavior)
    x_ref, res_ref = build_input(DATA, N, pixel_limit=255_000)
    print(f"ref resolution: {res_ref[0]}x{res_ref[1]}\n")
    poses_ref = forward_poses(model, x_ref, WIN, OV)

    for plim in [200_000, 160_000, 130_000, 100_000, 80_000, 60_000]:
        x_t, res = build_input(DATA, N, pixel_limit=plim)
        poses_t = forward_poses(model, x_t, WIN, OV)
        m = evaluate(poses_ref, poses_t)
        print(f"  res {res[0]:>3}x{res[1]:<3} ({res[0]*res[1]:>6,} px)  "
              f"ATE={m['ate_m']*1000:5.1f} mm  "
              f"trans-max={m['trans_max_m']*1000:5.1f} mm  "
              f"rot-RMSE={m['rot_rmse_deg']:.3f}deg  "
              f"rot-max={m['rot_max_deg']:.3f}deg  "
              f"scale={m['scale_align']:.4f}")


if __name__ == "__main__":
    main()
