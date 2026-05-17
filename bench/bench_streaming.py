"""Rigorous benchmark: StreamingLoGeR per-frame latency vs equivalent baseline.

Baseline ("naive sliding window"): for each new frame, call
    model.forward(latest_W_frames, window_size=W, overlap_size=O)
and extract the last pose. This re-encodes all W frames each call.

Streaming: StreamingLoGeR.update(new_frame) — encodes only the new frame,
reuses cached encoder features for the W-1 older frames, then runs the
decoder + heads on the same buffered window. With state persistence OFF
the output is bit-equivalent (modulo bf16 ordering) to the baseline.

Methodology mirrors bench_rigorous.py:
  - Phase 0: baseline vs baseline (placebo / noise floor)
  - Phase 1: baseline vs streaming
  - Interleaved measurements, 30 samples each, Welch's t-test, ratio CI
  - Skip the warmup frames (first W-1 calls): we time the steady-state.
"""

import sys, glob, inspect, yaml, math, time, csv, statistics
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.streaming import StreamingLoGeR, StreamingConfig


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


def welch_t_stats(a, b):
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sa, sb = va / na, vb / nb
    se = math.sqrt(sa + sb)
    if se == 0:
        return float("inf"), float("inf"), 0.0
    t = (ma - mb) / se
    dof = (sa + sb) ** 2 / ((sa ** 2) / (na - 1) + (sb ** 2) / (nb - 1))
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, dof, p


def ratio_ci(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    log_r = math.log(ma / mb)
    var_log_r = (va / (ma ** 2 * na)) + (vb / (mb ** 2 * nb))
    se = math.sqrt(var_log_r)
    return math.exp(log_r), math.exp(log_r - 1.96 * se), math.exp(log_r + 1.96 * se)


def fmt(name, times):
    n = len(times)
    m = statistics.mean(times) * 1000
    sd = statistics.stdev(times) * 1000 if n > 1 else 0.0
    sem = sd / math.sqrt(n)
    return (f"  {name:<22} n={n}  mean={m:7.2f} ms  ±{sd:5.2f}sd  ±{sem:5.2f}sem  "
            f"-> {1000/m:5.2f} FPS")


def time_baseline_update(model, buffer, window, overlap):
    """One 'frame arrives' tick under the naive sliding-window baseline."""
    x = torch.stack(buffer[-window:]).unsqueeze(0).to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=window, overlap_size=overlap)
    torch.cuda.synchronize()
    _ = out["camera_poses"][0, -1]
    return time.perf_counter() - t0


def time_streaming_update(streamer, frame):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = streamer.update(frame.to("cuda"))
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("[setup] loading model ...", flush=True)
    t0 = time.time()
    model = load_model()
    print(f"[setup] loaded in {time.time()-t0:.1f}s\n")

    WIN, OV = 8, 3
    N_FRAMES = 50  # need enough to keep cycling
    N_SAMPLES = 30  # samples per stack

    print("[data] building frames ...", flush=True)
    frames, (TW, TH) = build_frames("data/examples/office", N_FRAMES)
    print(f"[data] {N_FRAMES} frames @ {TW}x{TH}\n")

    # ===== Build streamer (with compile ON, since that's the deploy config) =====
    print("[stream] preparing streamer with compile=True ...", flush=True)
    streamer = StreamingLoGeR(model, StreamingConfig(
        window_size=WIN, overlap_size=OV, compile=True, persist_state=False))
    # Warm up the streamer (encoder compile + warmup the decode path)
    print("  warming streamer (first compile pass may take 30-60s)...", flush=True)
    t_warm = time.time()
    for fr in frames[:WIN]:
        _ = streamer.update(fr.to("cuda"))
    # A few more to flush
    for fr in frames[WIN:WIN+3]:
        _ = streamer.update(fr.to("cuda"))
    print(f"  warmup done in {time.time()-t_warm:.1f}s\n")

    # Warm up baseline too
    print("[baseline] warming ...", flush=True)
    for _ in range(3):
        _ = time_baseline_update(model, frames[:WIN], WIN, OV)
    print("  done\n")

    # ===== Phase 0: noise floor (baseline vs baseline) =====
    print("=== Phase 0: PLACEBO  baseline vs baseline ===", flush=True)
    p0_a, p0_b = [], []
    for i in range(N_SAMPLES * 2):
        # cycle the buffer end so we don't trivially get the same input every time
        start = i % (N_FRAMES - WIN)
        buf = frames[start:start + WIN]
        t = time_baseline_update(model, buf, WIN, OV)
        (p0_a if i % 2 == 0 else p0_b).append(t)
    print(fmt("baseline-A", p0_a))
    print(fmt("baseline-B", p0_b))
    t, dof, pv = welch_t_stats(p0_a, p0_b)
    r0, r0_lo, r0_hi = ratio_ci(p0_a, p0_b)
    print(f"  welch t={t:.2f} dof={dof:.0f} p={pv:.3f}  "
          f"ratio A/B={r0:.3f} 95%CI [{r0_lo:.3f}, {r0_hi:.3f}]")

    # ===== Phase 1: baseline vs streaming =====
    print("\n=== Phase 1: baseline vs StreamingLoGeR ===", flush=True)
    # Restart streamer to start at known state (don't reset every iter)
    streamer.reset()
    # re-prime with WIN frames first
    for fr in frames[:WIN]:
        _ = streamer.update(fr.to("cuda"))

    p1_a, p1_b = [], []
    for i in range(N_SAMPLES * 2):
        if i % 2 == 0:
            # baseline tick: simulate a new frame arriving; window is the latest W
            start = i % (N_FRAMES - WIN)
            buf = frames[start:start + WIN]
            t = time_baseline_update(model, buf, WIN, OV)
            p1_a.append(t)
        else:
            # streaming tick: feed the next frame in sequence (cycling)
            idx = (i // 2 + WIN) % N_FRAMES
            t = time_streaming_update(streamer, frames[idx])
            p1_b.append(t)
    print(fmt("baseline", p1_a))
    print(fmt("streaming", p1_b))
    t, dof, pv = welch_t_stats(p1_a, p1_b)
    r1, r1_lo, r1_hi = ratio_ci(p1_a, p1_b)
    sig = "SIGNIFICANT" if pv < 0.05 else "NOT significant"
    print(f"  welch t={t:.2f} dof={dof:.0f} p={pv:.2e} [{sig}]")
    print(f"  ratio baseline/streaming = {r1:.3f}  95%CI [{r1_lo:.3f}, {r1_hi:.3f}]")
    base_fps = 1 / statistics.mean(p1_a)
    stream_fps = 1 / statistics.mean(p1_b)
    print(f"  baseline FPS: {base_fps:.2f}   streaming FPS: {stream_fps:.2f}   "
          f"({stream_fps - base_fps:+.2f} FPS)")

    # Save raw
    out = REPO / "bench" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "streaming_phase0_placebo.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stack", "seconds"])
        for x in p0_a: w.writerow(("baseA", x))
        for x in p0_b: w.writerow(("baseB", x))
    with open(out / "streaming_phase1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stack", "seconds"])
        for x in p1_a: w.writerow(("baseline", x))
        for x in p1_b: w.writerow(("streaming", x))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  placebo ratio = {r0:.3f}  95%CI [{r0_lo:.3f}, {r0_hi:.3f}]  (noise floor)")
    print(f"  streaming speedup = {r1:.3f}  95%CI [{r1_lo:.3f}, {r1_hi:.3f}]  ({sig})")
    real = (r1_lo > r0_hi)
    if real:
        print(f"  -> CI does NOT overlap placebo: REAL speedup")
    else:
        print(f"  -> CI overlaps placebo: NOT a real speedup")


if __name__ == "__main__":
    main()
