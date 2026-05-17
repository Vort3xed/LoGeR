"""Resolution sweep: measure latency vs input image size at fixed window."""

import sys, glob, inspect, yaml, torch, time, math
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3


def load_model(ckpt, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in sig.parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    kwargs = {k: cfg[k] for k in cfg if k in valid}
    model = Pi3(**kwargs)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to("cuda")


def res_from_limit(W_native, H_native, pixel_limit):
    scale = math.sqrt(pixel_limit / (W_native * H_native))
    Wt, Ht = W_native * scale, H_native * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def build_input(image_dir, n, TW, TH):
    paths = natsorted(glob.glob(f"{image_dir}/*.png"))[:n]
    out = torch.empty((len(paths), 3, TH, TW), dtype=torch.float32)
    to_t = transforms.ToTensor()
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            out[i].copy_(to_t(img.convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS)))
    return out.unsqueeze(0).to("cuda")


def bench(model, x, window, overlap, runs=5, warmups=2):
    fwd_kwargs = dict(window_size=window, overlap_size=overlap)
    for _ in range(warmups):
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, **fwd_kwargs)
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, **fwd_kwargs)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def stat(ts):
    n = len(ts)
    m = sum(ts) / n
    s = (sum((t-m)**2 for t in ts)/n) ** 0.5
    return m, s, min(ts), max(ts)


def main():
    DATA = "data/examples/office"
    print("[setup] loading model...", flush=True)
    t0 = time.time()
    model = load_model("ckpts/LoGeR/latest.pt", "ckpts/LoGeR/original_config.yaml")
    print(f"[setup] loaded in {time.time()-t0:.1f}s\n")

    # Office native: 640x480
    paths = sorted(glob.glob(f"{DATA}/*.png"))
    with Image.open(paths[0]) as im:
        Wn, Hn = im.size

    print("=" * 76)
    print(f"  native image: {Wn}x{Hn}  |  window=8, overlap=3")
    print("=" * 76)

    # Resolutions to test (kept as DINOv2 patch multiples of 14)
    pixel_limits = [255_000, 160_000, 100_000, 70_000]
    rows = []
    for plim in pixel_limits:
        TW, TH = res_from_limit(Wn, Hn, plim)
        x = build_input(DATA, 8, TW, TH)
        print(f"--- {TW}x{TH} ({TW*TH:,} px, limit={plim:,}) ---", flush=True)
        ts = bench(model, x, window=8, overlap=3, runs=5, warmups=2)
        m, s, lo, hi = stat(ts)
        rows.append((TW, TH, TW*TH, m, s))
        print(f"  baseline:  mean={m*1000:7.1f} ms  ±{s*1000:.1f}   "
              f"-> {8/m:5.2f} FPS  ({m/8*1000:.1f} ms/frame)", flush=True)
        # compiled
        torch.cuda.empty_cache()
        compiled = torch.compile(model, mode="default", dynamic=False, fullgraph=False)
        t_c = time.time()
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            _ = compiled(x, window_size=8, overlap_size=3)
        torch.cuda.synchronize()
        print(f"  (compile first call: {time.time()-t_c:.1f}s)")
        ts_c = bench(compiled, x, window=8, overlap=3, runs=5, warmups=2)
        m_c, s_c, _, _ = stat(ts_c)
        rows.append((TW, TH, TW*TH, m_c, s_c))
        print(f"  compiled:  mean={m_c*1000:7.1f} ms  ±{s_c*1000:.1f}   "
              f"-> {8/m_c:5.2f} FPS  ({m_c/8*1000:.1f} ms/frame)   "
              f"speedup-vs-baseline: {m/m_c:.2f}x", flush=True)
        del compiled
        torch.cuda.empty_cache()
        print()

    print("\npeak VRAM MB:", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
