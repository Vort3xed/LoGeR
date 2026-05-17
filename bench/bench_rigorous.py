"""Rigorous A/B benchmark of LoGeR optimizations.

Key methodological choices to avoid spurious "speedups":

1) INTERLEAVED measurement: we don't run "all A then all B". Instead we cycle
   A,B,A,B,...  Drift in GPU temperature, clock throttling, or memory-state
   shifts hits both stacks symmetrically.

2) PLACEBO: we run baseline vs baseline (A vs A) first to measure noise floor.
   If A vs A shows ~10% gap, the bench is unreliable and reported speedups
   shouldn't be trusted.

3) Enough samples for stats: 30 per stack (after warmup). Welch's t-test for
   significance; report mean +- standard error of the mean, and the 95% CI for
   the speedup ratio via the delta method.

4) Per-call traces are saved to bench/results/*.csv so you can inspect raw
   measurements.

5) Fixed input tensor: we build it once, place on GPU, and reuse the same
   memory across both stacks.  No data-loading variance.
"""

import sys, glob, inspect, yaml, torch, time, math, csv, statistics, os
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
    return out.unsqueeze(0).to("cuda"), (TW, TH)


def single_call_time(model, x, window, overlap):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(x, window_size=window, overlap_size=overlap)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def warmup(model, x, window, overlap, n=3):
    for _ in range(n):
        _ = single_call_time(model, x, window, overlap)


def welch_t_stats(a, b):
    """Return (t-stat, dof, two-sided-p-approx)."""
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sa, sb = va / na, vb / nb
    se = math.sqrt(sa + sb)
    if se == 0:
        return float("inf"), float("inf"), 0.0
    t = (ma - mb) / se
    dof = (sa + sb) ** 2 / ((sa ** 2) / (na - 1) + (sb ** 2) / (nb - 1))
    # crude two-sided p from normal approx (dof is large here)
    z = abs(t)
    # erfc-based: p ~= erfc(z/sqrt(2))
    p = math.erfc(z / math.sqrt(2.0))
    return t, dof, p


def ratio_ci(a, b):
    """95% CI for mean(a)/mean(b) via delta method (log-ratio)."""
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    if ma <= 0 or mb <= 0:
        return None
    log_r = math.log(ma / mb)
    var_log_r = (va / (ma ** 2 * na)) + (vb / (mb ** 2 * nb))
    se = math.sqrt(var_log_r)
    lo = math.exp(log_r - 1.96 * se)
    hi = math.exp(log_r + 1.96 * se)
    return math.exp(log_r), lo, hi


def fmt_stack(name, times):
    n = len(times)
    m = statistics.mean(times) * 1000
    sd = statistics.stdev(times) * 1000 if n > 1 else 0.0
    sem = sd / math.sqrt(n)
    lo = min(times) * 1000
    hi = max(times) * 1000
    return f"  {name:<18} n={n}  mean={m:7.2f} ms  ±{sd:5.2f}sd  ±{sem:5.2f}sem  [{lo:.1f} .. {hi:.1f}]"


def interleaved_bench(label_A, model_A, label_B, model_B, x, window, overlap, n_per=30):
    """Run A,B,A,B,...  Return (times_A, times_B, order_log)."""
    times_A, times_B, order = [], [], []
    total = n_per * 2
    for i in range(total):
        if i % 2 == 0:
            t = single_call_time(model_A, x, window, overlap)
            times_A.append(t)
            order.append((i, label_A, t))
        else:
            t = single_call_time(model_B, x, window, overlap)
            times_B.append(t)
            order.append((i, label_B, t))
    return times_A, times_B, order


def main():
    DATA = "data/examples/office"
    OUT_DIR = REPO / "bench" / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print(f"  GPU: {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}  "
          f"torch {torch.__version__}")
    print(f"  power state: ", end="")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw",
             "--format=csv,noheader,nounits"]
        ).decode().strip()
        print(out)
    except Exception:
        print("(nvidia-smi unavailable)")
    print("=" * 86)
    print("\n[setup] loading model ...", flush=True)
    t0 = time.time()
    base_model = load_model("ckpts/LoGeR/latest.pt", "ckpts/LoGeR/original_config.yaml")
    print(f"[setup] loaded in {time.time()-t0:.1f}s\n", flush=True)

    WINDOW = 8
    OVERLAP = 3
    N_PER_STACK = 30

    x, (TW, TH) = build_input(DATA, WINDOW)
    print(f"input: window={WINDOW} overlap={OVERLAP}  shape={tuple(x.shape)} ({TW}x{TH})\n",
          flush=True)

    # ===== Phase 0: noise floor =====
    print("\n=== Phase 0: PLACEBO (baseline vs baseline)  — measures bench noise floor ===\n",
          flush=True)
    warmup(base_model, x, WINDOW, OVERLAP, n=5)
    p0_a, p0_b, order0 = interleaved_bench("baseA", base_model, "baseB", base_model, x,
                                            WINDOW, OVERLAP, n_per=N_PER_STACK)
    print(fmt_stack("baseline-A", p0_a))
    print(fmt_stack("baseline-B", p0_b))
    t, dof, p_val = welch_t_stats(p0_a, p0_b)
    print(f"  welch t={t:.2f} dof={dof:.0f} p={p_val:.3f}  "
          f"(if p>0.05 the two runs are statistically indistinguishable)")
    r0, r0_lo, r0_hi = ratio_ci(p0_a, p0_b)
    print(f"  speedup-ratio A/B = {r0:.3f}  [{r0_lo:.3f}, {r0_hi:.3f}]  (95% CI)")
    print(f"  -> NOISE FLOOR: any reported speedup within [{r0_lo:.3f}, {r0_hi:.3f}] "
          f"is indistinguishable from chance.")

    with open(OUT_DIR / "phase0_placebo.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "label", "seconds"])
        for r_ in order0:
            w.writerow(r_)

    # ===== Phase 1: baseline vs torch.compile (default) =====
    print("\n=== Phase 1: baseline vs torch.compile(mode='default') ===\n", flush=True)
    compiled = torch.compile(base_model, mode="default", dynamic=False, fullgraph=False)
    print("  warming compiled (this triggers compilation)...", flush=True)
    t_c = time.time()
    warmup(compiled, x, WINDOW, OVERLAP, n=5)
    print(f"  compile + warmup: {time.time()-t_c:.1f}s\n")

    p1_a, p1_b, order1 = interleaved_bench("baseline", base_model, "compile", compiled, x,
                                            WINDOW, OVERLAP, n_per=N_PER_STACK)
    print(fmt_stack("baseline", p1_a))
    print(fmt_stack("compile", p1_b))
    t, dof, p_val = welch_t_stats(p1_a, p1_b)
    r1, r1_lo, r1_hi = ratio_ci(p1_a, p1_b)
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    print(f"  welch t={t:.2f} dof={dof:.0f} p={p_val:.2e}  [{sig}]")
    print(f"  speedup ratio baseline/compile = {r1:.3f}  95%CI [{r1_lo:.3f}, {r1_hi:.3f}]")

    with open(OUT_DIR / "phase1_baseline_vs_compile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "label", "seconds"])
        for r_ in order1:
            w.writerow(r_)

    # ===== Phase 2: baseline vs (compile + lower resolution) =====
    print("\n=== Phase 2: baseline vs (compile + smaller resolution) ===\n", flush=True)
    x_lo, (TW2, TH2) = build_input(DATA, WINDOW, pixel_limit=100_000)
    print(f"  small input: {tuple(x_lo.shape)}  ({TW2}x{TH2}, "
          f"{(TW2*TH2)/(TW*TH):.2f}x pixel count)")
    # Re-warmup compile for new shape
    print("  warming compile for low-res ...", flush=True)
    t_c = time.time()
    warmup(compiled, x_lo, WINDOW, OVERLAP, n=5)
    print(f"  compile + warmup: {time.time()-t_c:.1f}s\n")

    p2_a, p2_b, order2 = interleaved_bench("baseline-full", base_model, "compile-low",
                                            compiled, x, WINDOW, OVERLAP, n_per=N_PER_STACK)
    # Above is comparing same-resolution baseline vs same-resolution compile — already covered.
    # We want: full-res-baseline vs low-res-compile. Different inputs.
    # Build a wrapper that always uses x_lo for "fast" stack and x for "baseline".
    def baseline_call(_x):
        return single_call_time(base_model, x, WINDOW, OVERLAP)

    def fast_call(_x):
        return single_call_time(compiled, x_lo, WINDOW, OVERLAP)

    # Custom interleave with different inputs
    p2_a, p2_b, order2 = [], [], []
    warmup(base_model, x, WINDOW, OVERLAP, n=3)
    warmup(compiled, x_lo, WINDOW, OVERLAP, n=3)
    for i in range(N_PER_STACK * 2):
        if i % 2 == 0:
            t = single_call_time(base_model, x, WINDOW, OVERLAP)
            p2_a.append(t); order2.append((i, "baseline-full-res", t))
        else:
            t = single_call_time(compiled, x_lo, WINDOW, OVERLAP)
            p2_b.append(t); order2.append((i, "compile-low-res", t))

    print(fmt_stack("baseline (574x434)", p2_a))
    print(fmt_stack(f"compile ({TW2}x{TH2})", p2_b))
    t, dof, p_val = welch_t_stats(p2_a, p2_b)
    r2, r2_lo, r2_hi = ratio_ci(p2_a, p2_b)
    sig2 = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    print(f"  welch t={t:.2f} dof={dof:.0f} p={p_val:.2e}  [{sig2}]")
    print(f"  speedup ratio = {r2:.3f}  95%CI [{r2_lo:.3f}, {r2_hi:.3f}]")
    print(f"  baseline FPS: {WINDOW/statistics.mean(p2_a):.2f}    "
          f"fast FPS: {WINDOW/statistics.mean(p2_b):.2f}")

    with open(OUT_DIR / "phase2_baseline_vs_compile_lowres.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "label", "seconds"])
        for r_ in order2:
            w.writerow(r_)

    # ===== Final summary =====
    print("\n" + "=" * 86)
    print("  SUMMARY")
    print("=" * 86)
    print(f"  Phase 0 (PLACEBO baseline vs baseline):")
    print(f"     ratio A/B = {r0:.3f}  95%CI [{r0_lo:.3f}, {r0_hi:.3f}]")
    print(f"     -> noise floor: speedups in this CI are indistinguishable from chance")
    print(f"  Phase 1 (baseline vs torch.compile, same res):")
    print(f"     ratio baseline/compile = {r1:.3f}  95%CI [{r1_lo:.3f}, {r1_hi:.3f}]   ({sig})")
    if r1_lo > r0_hi:
        print(f"     -> compile speedup CI does NOT overlap placebo CI: REAL speedup")
    else:
        print(f"     -> compile speedup CI overlaps placebo CI: NOT a real speedup")
    print(f"  Phase 2 (full-res baseline vs compile + low-res):")
    print(f"     ratio = {r2:.3f}  95%CI [{r2_lo:.3f}, {r2_hi:.3f}]   ({sig2})")
    if r2_lo > r0_hi:
        print(f"     -> combined speedup CI does NOT overlap placebo CI: REAL")
    else:
        print(f"     -> combined speedup CI overlaps placebo CI: NOT real")
    print()
    print("peak VRAM (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
