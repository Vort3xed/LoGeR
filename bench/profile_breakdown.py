"""Time-breakdown profile of a LoGeR forward pass.

Splits the forward into:
  - encoder (DINOv2, per-frame, embarrassingly parallel)
  - decoder (windowed cross-frame attention + TTT/SWA)
  - heads (camera pose + local points + conf)

Helps decide where optimization effort pays off most.
"""

import sys, glob, inspect, yaml, torch, time, math
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3


def load_model(ckpt_path, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {
        n for n, p in sig.parameters.items()
        if n not in {"self", "args", "kwargs"}
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    kwargs = {k: cfg[k] for k in cfg if k in valid}
    model = Pi3(**kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to("cuda")


def build_input(image_dir, n, pixel_limit=255_000):
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
    return out.unsqueeze(0).to("cuda")


def timed(label, fn, runs=5, warmup=2):
    for _ in range(warmup):
        torch.cuda.synchronize(); fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    m = sum(ts) / len(ts)
    print(f"  {label:>16}  mean = {m*1000:7.1f} ms  (min {min(ts)*1000:.1f}, max {max(ts)*1000:.1f})")
    return m


def main():
    ckpt = "ckpts/LoGeR/latest.pt"
    cfg = "ckpts/LoGeR/original_config.yaml"
    data = "data/examples/office"

    print("[setup] loading model ...", flush=True)
    t0 = time.time()
    model = load_model(ckpt, cfg)
    print(f"[setup] loaded in {time.time()-t0:.1f}s\n")

    for N in [8, 16, 32]:
        x = build_input(data, N)
        B, N_, C, H, W = x.shape
        assert N_ == N
        print(f"=== N={N}  input {tuple(x.shape)} ===")

        # imgs preproc that forward() does internally
        x_dev = x.to("cuda")
        x_norm = (x_dev - model.image_mean) / model.image_std
        x_flat = x_norm.reshape(B * N, C, H, W)

        # ---- encoder only ----
        def run_encoder():
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                h = model.encoder(x_flat, is_training=True)
                if isinstance(h, dict):
                    h = h["x_norm_patchtokens"]
            return h

        enc_t = timed("encoder", lambda: run_encoder())

        # ---- decoder + heads together (full forward minus our explicit preproc + encoder) ----
        # Easiest fair measurement: time full forward, subtract encoder.
        def run_full():
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(x, window_size=N, overlap_size=3)

        full_t = timed("full forward", run_full)
        rest_t = full_t - enc_t
        enc_pct = enc_t / full_t * 100
        rest_pct = rest_t / full_t * 100
        print(f"   -> encoder: {enc_pct:4.1f}%   decoder+heads+misc: {rest_pct:4.1f}%   "
              f"({enc_t*1000:.1f} + {rest_t*1000:.1f} ms)\n")

    print("peak VRAM (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
