"""Sweep streaming latency + accuracy across window sizes.

For each W in {2, 4, 6, 8}:
  - bench: per-frame update latency under StreamingLoGeR(compile=True)
  - accuracy: pose for the latest frame vs W=8 reference, over a sliding window
"""

import sys, glob, inspect, yaml, math, time, statistics, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
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


def pose_diff(p_ref, p_test):
    td = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    Re = p_ref[:3, :3].T @ p_test[:3, :3]
    c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
    rd = math.degrees(math.acos(c))
    return td, rd


def main():
    print("[setup] loading model ...", flush=True)
    model = load_model()
    N = 40
    frames, _ = build_frames("data/examples/office", N)
    print(f"[data] {N} frames\n")

    # ===== Reference: W=8 batch model.forward at each sliding window =====
    print("[ref] computing W=8 batch poses per sliding window ...", flush=True)
    REF_W = 8
    ref_poses = {}
    for T in range(REF_W - 1, N):
        win = torch.stack(frames[T - REF_W + 1:T + 1]).unsqueeze(0).to("cuda")
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(win, window_size=REF_W, overlap_size=3)
        ref_poses[T] = out["camera_poses"][0, -1].float().cpu().numpy()
    print(f"  collected {len(ref_poses)} reference poses\n")

    # ===== For each W: latency + accuracy =====
    print(f"{'W':>3}  {'mean ms':>9}  {'FPS':>6}  {'rot RMSE':>10}  {'rot max':>9}  {'t RMSE mm':>10}")
    print("-" * 70)
    results = []
    for W_TEST in [2, 4, 6, 8]:
        OV_TEST = max(W_TEST - 1, 0) // 2  # pick a small overlap
        streamer = StreamingLoGeR(model, StreamingConfig(
            window_size=W_TEST, overlap_size=OV_TEST, compile=True, persist_state=False))
        # Warm up
        for fr in frames[:W_TEST + 3]:
            _ = streamer.update(fr.to("cuda"))

        # Reset to clean state, then collect both timing + poses
        streamer.reset()
        for fr in frames[:W_TEST - 1]:
            _ = streamer.update(fr.to("cuda"))

        times = []
        stream_poses = {}
        for i in range(W_TEST - 1, N):
            fr = frames[i].to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = streamer.update(fr)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if p is not None:
                stream_poses[i] = p.numpy()

        m = statistics.mean(times) * 1000
        fps = 1000 / m
        # Compare against ref where both have a pose (T >= REF_W - 1 AND T >= W_TEST - 1)
        rds, tds = [], []
        for T, ref_p in ref_poses.items():
            if T in stream_poses:
                td, rd = pose_diff(ref_p, stream_poses[T])
                tds.append(td); rds.append(rd)
        rot_rmse = math.sqrt(sum(r**2 for r in rds)/len(rds)) if rds else 0
        rot_max = max(rds) if rds else 0
        t_rmse = math.sqrt(sum(t**2 for t in tds)/len(tds))*1000 if tds else 0
        print(f"  {W_TEST:>3}  {m:8.1f}  {fps:5.2f}  {rot_rmse:9.3f}°  {rot_max:8.3f}°  {t_rmse:9.2f}")
        results.append((W_TEST, m, fps, rot_rmse, rot_max, t_rmse))
        del streamer
        torch.cuda.empty_cache()

    print("\nLEGEND: 'rot RMSE' is vs W=8 batch model.forward output for the same window.")
    print("       Lower W means less long-context info; the model still produces useful poses.")


if __name__ == "__main__":
    main()
