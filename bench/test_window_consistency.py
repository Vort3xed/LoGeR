"""Are two fresh model.forward() calls on overlapping windows Sim(3)-relatable?

If yes -> our stitching approach should work, I have a bug somewhere.
If no  -> the model's internal state is doing more than just Sim(3) alignment,
         and we need a different stitching strategy.
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
from loger.trajectory import umeyama_sim3, apply_sim3_to_pose, align_poses_sim3


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
    print("[setup] loading ...", flush=True)
    model = load_model()
    frames, _ = build_frames("data/examples/office", 16)
    WIN = 8
    OV = 3

    def fresh_forward(start, end):
        x = torch.stack(frames[start:end]).unsqueeze(0).to("cuda")
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, window_size=WIN, overlap_size=OV)
        return out["camera_poses"][0].float().cpu().numpy()  # (W, 4, 4)

    print("\n--- Two consecutive overlapping windows ---")
    poses_A = fresh_forward(0, 8)   # frames 0..7
    poses_B = fresh_forward(1, 9)   # frames 1..8
    # Overlap: frame indices 1..7 in A correspond to indices 0..6 in B
    pts_A = poses_A[1:8, :3, 3]   # (7, 3)
    pts_B = poses_B[0:7, :3, 3]   # (7, 3)

    print(f"\nRaw translation diffs at overlap frames:")
    for i in range(7):
        td, rd = pose_diff(poses_A[1 + i], poses_B[0 + i])
        print(f"  frame {1+i}: A={poses_A[1+i,:3,3]}  B={poses_B[0+i,:3,3]}   "
              f"t-diff={td*1000:.2f} mm  rot-diff={rd:.3f}°")

    # Old translation-only fit (for comparison)
    print("\n--- Fit Sim(3) A -> B on translations ONLY (old method) ---")
    scale, R, t = umeyama_sim3(pts_A, pts_B)
    rds, tds = [], []
    for i in range(7):
        p_A_aligned = apply_sim3_to_pose(poses_A[1 + i], scale, R, t)
        td, rd = pose_diff(poses_B[0 + i], p_A_aligned)
        rds.append(rd); tds.append(td)
    print(f"  rot RMSE residual: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.4f}°    "
          f"t RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.3f} mm")

    # New: full-pose Sim(3) fit using both translations AND rotations
    print("\n--- Fit Sim(3) A -> B using FULL POSES (rotations + translations) ---")
    src = poses_A[1:8]
    dst = poses_B[0:7]
    scale, R, t = align_poses_sim3(src, dst)
    print(f"  scale: {scale:.4f}")
    print(f"  R diag: {np.diag(R)}    R off-norm: {np.linalg.norm(R - np.eye(3)):.4f}")
    print(f"  t: {t}    |t|: {np.linalg.norm(t):.4f}")
    rds, tds = [], []
    for i in range(7):
        p_A_aligned = apply_sim3_to_pose(poses_A[1 + i], scale, R, t)
        td, rd = pose_diff(poses_B[0 + i], p_A_aligned)
        rds.append(rd); tds.append(td)
        print(f"  frame {1+i}: residual t-diff={td*1000:.3f} mm  rot-diff={rd:.4f}°")
    print(f"\n  rot RMSE residual: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.4f}°    "
          f"t RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.3f} mm")
    if max(rds) < 1.0 and max(tds) < 0.005:
        print("  -> Two overlapping windows ARE related by a Sim(3). Stitching is feasible.")
    else:
        print("  -> Two overlapping windows are NOT cleanly Sim(3)-related.")


if __name__ == "__main__":
    main()
