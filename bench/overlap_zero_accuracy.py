"""Quick check: how much does overlap_size=0 (uniform PE token) vs overlap_size=3
change LoGeR's output per single-window call?

This matters because incremental decode with K/V caching is only valid when each
frame's PE classification stays constant as the window slides — which requires
overlap_size=0 (so every frame gets pe_token_1, the "other" PE).
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


def main():
    print("[setup] loading ...", flush=True)
    model = load_model()
    WIN = 8
    N = 16
    frames, _ = build_frames("data/examples/office", N)
    print(f"[data] {N} frames\n")

    # Reference: overlap=3 (current demo default)
    # Compare overlap in {0, 1, 2, 4, 5, 7} for last-of-window pose
    print("Sweep overlap_size — measuring accuracy vs overlap=3 reference (last-of-window pose):")
    for OV_TEST in [4, 5, 6, 7, 8, 9]:
        rot_diffs, t_diffs = [], []
        for T in range(WIN - 1, N):
            win = torch.stack(frames[T - WIN + 1:T + 1]).unsqueeze(0).to("cuda")
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                out3 = model(win, window_size=WIN, overlap_size=3)
                out_t = model(win, window_size=WIN, overlap_size=OV_TEST)
            p3 = out3["camera_poses"][0, -1].float().cpu().numpy()
            pt = out_t["camera_poses"][0, -1].float().cpu().numpy()
            td = float(np.linalg.norm(p3[:3, 3] - pt[:3, 3]))
            Re = p3[:3, :3].T @ pt[:3, :3]
            cos = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
            rd = math.degrees(math.acos(cos))
            rot_diffs.append(rd); t_diffs.append(td)
        print(f"  overlap={OV_TEST}:  rot mean={sum(rot_diffs)/len(rot_diffs):.3f} deg "
              f"max={max(rot_diffs):.3f}    "
              f"t mean={sum(t_diffs)/len(t_diffs)*1000:.2f} mm "
              f"max={max(t_diffs)*1000:.2f}")


if __name__ == "__main__":
    main()
