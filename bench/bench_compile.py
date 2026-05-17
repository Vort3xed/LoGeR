"""Benchmark LoGeR forward latency under different optimization stacks.

Tests these stacks (each cumulative or independent, see CLI):
  - baseline (fp32 weights + bf16 autocast, the demo's default)
  - +tf32  (enable TF32 for fp32 matmuls — cheap free win for fp32 ops)
  - +bf16weights (cast model weights to bf16 — saves memory bandwidth)
  - +compile (torch.compile mode="default", no CUDA graphs since SVD breaks them)

Goal: measure pure inference latency for a fixed-shape window — the metric
that matters for real-time pose tracking.
"""

import sys, glob, inspect, yaml, torch, time, math, argparse, copy
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3  # noqa: E402


def load_model(ckpt_path: str, cfg_path: str) -> Pi3:
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


def build_input(image_dir: str, n: int, pixel_limit: int = 255_000):
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


def bench(model, x, window, overlap, runs: int = 6, warmups: int = 2):
    fwd_kwargs = dict(window_size=window, overlap_size=overlap)
    for _ in range(warmups):
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, **fwd_kwargs)
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, **fwd_kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def stat(times):
    n = len(times)
    mean = sum(times) / n
    var = sum((t - mean) ** 2 for t in times) / n
    return mean, var ** 0.5, min(times), max(times)


def fmt(label, win, times):
    m, s, lo, hi = stat(times)
    fps = win / m
    return (
        f"{label:>26}  win={win:>2}  "
        f"mean={m*1000:7.1f} ms  ±{s*1000:5.1f}  "
        f"[{lo*1000:6.1f}, {hi*1000:6.1f}]  "
        f"-> {fps:5.2f} FPS  ({m/win*1000:5.1f} ms/frame)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/LoGeR/latest.pt")
    ap.add_argument("--cfg", default="ckpts/LoGeR/original_config.yaml")
    ap.add_argument("--data", default="data/examples/office")
    ap.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--stacks", nargs="+",
                    default=["baseline", "tf32", "bf16w", "tf32+bf16w", "tf32+bf16w+compile"])
    args = ap.parse_args()

    print("=" * 86)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}  "
          f"torch {torch.__version__}")
    print("=" * 86)

    print("\n[setup] loading model...", flush=True)
    t0 = time.time()
    base_model = load_model(args.ckpt, args.cfg)
    print(f"[setup] loaded in {time.time()-t0:.1f}s  "
          f"params={sum(p.numel() for p in base_model.parameters())/1e6:.1f}M\n", flush=True)

    results = {}  # (stack, win) -> times

    def reset_global_state():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    for stack in args.stacks:
        print(f"=== stack: {stack} ===", flush=True)
        reset_global_state()
        if "tf32" in stack:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            print("  + TF32 enabled")
        if "bf16w" in stack:
            print("  + casting model weights to bf16...")
            model = copy.deepcopy(base_model)
            for p in model.parameters():
                p.data = p.data.to(torch.bfloat16)
            for b in model.buffers():
                if b.dtype == torch.float32:
                    b.data = b.data.to(torch.bfloat16)
        else:
            model = base_model

        if "compile" in stack:
            print("  + torch.compile(mode='default', dynamic=False)")
            model = torch.compile(model, mode="default", dynamic=False, fullgraph=False)

        for win in args.windows:
            x, (TW, TH) = build_input(args.data, win)
            print(f"  [{stack}] win={win}  input={tuple(x.shape)} ...", flush=True)
            if "compile" in stack:
                t_c = time.time()
                with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x, window_size=win, overlap_size=args.overlap)
                torch.cuda.synchronize()
                print(f"    (first call w/ compile: {time.time()-t_c:.1f}s)", flush=True)
            results[(stack, win)] = bench(model, x, win, args.overlap,
                                          runs=args.runs, warmups=args.warmups)
            print("  " + fmt(stack, win, results[(stack, win)]), flush=True)

        if "bf16w" in stack:
            del model
            torch.cuda.empty_cache()
        print()

    print("=" * 86)
    print("  SUMMARY  (per-window mean wall time)")
    print("=" * 86)
    for win in args.windows:
        base_mean = stat(results[("baseline", win)])[0] if ("baseline", win) in results else None
        for stack in args.stacks:
            if (stack, win) not in results:
                continue
            m, s, lo, hi = stat(results[(stack, win)])
            fps = win / m
            line = (
                f"  {stack:<26}  win={win:>2}  "
                f"mean={m*1000:7.1f} ms  -> {fps:5.2f} FPS  ({m/win*1000:5.1f} ms/frame)"
            )
            if base_mean is not None and stack != "baseline":
                line += f"   speedup vs baseline: {base_mean/m:5.2f}x"
            print(line)
        print()

    print("peak VRAM (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
