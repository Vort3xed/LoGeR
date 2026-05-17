"""Stress test the pipeline on 1500 frames of real walking, with optional
live Viser visualization.

Usage:
    python bench/stress_and_viz.py            # headless stress test
    python bench/stress_and_viz.py --viz      # also start Viser viz at port 8080
    python bench/stress_and_viz.py --viz --camera-fps 30 --frames 1500
"""

import sys, glob, inspect, yaml, math, time, statistics, argparse, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.realtime_pipeline import RealtimePipeline, PipelineConfig
from loger.trajectory import align_poses_sim3, apply_sim3_to_pose


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
    return ([to_t(Image.open(p).convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS))
             for p in paths], paths, (TW, TH))


def parse_tum_groundtruth(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = (float(p) for p in parts[1:4])
            qx, qy, qz, qw = (float(p) for p in parts[4:8])
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
            ])
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
            out.append((ts, T))
    return out


def match_gt_to_frames(rgb_paths, gt):
    rgb_ts = [float(Path(p).stem) for p in rgb_paths]
    gt_ts = np.array([g[0] for g in gt])
    out = []
    for rt in rgb_ts:
        i = int(np.argmin(np.abs(gt_ts - rt)))
        out.append(gt[i][1])
    return out


def pose_diff(p_ref, p_test):
    td = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    Re = p_ref[:3, :3].T @ p_test[:3, :3]
    c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
    return td, math.degrees(math.acos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/rgbd_dataset_freiburg3_long_office_household")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--viz", action="store_true", help="Run live Viser viz")
    ap.add_argument("--viz-port", type=int, default=8080)
    ap.add_argument("--keep-alive", type=int, default=600,
                    help="After simulation, keep the viz server up this many seconds (default 600 = 10 min)")
    args = ap.parse_args()

    print(f"[setup] loading model + {args.frames} frames from {args.data} ...", flush=True)
    model = load_model()
    frames, paths, (TW, TH) = build_frames(f"{args.data}/rgb", args.frames)
    print(f"[data] {len(frames)} frames @ {TW}x{TH}\n", flush=True)

    pipeline = RealtimePipeline(model, PipelineConfig(
        window_size=8, overlap_size=3, compile=True, inference_queue_size=1))

    viz = None
    if args.viz:
        from loger.visualizer import TrajectoryVisualizer
        viz = TrajectoryVisualizer(port=args.viz_port)
        url = viz.start()
        print(f"[viz] open {url} in a browser to watch the trajectory grow.\n")
        # Wire up the worker → viz callback so the green "anchor" trail
        # gets populated whenever a real model inference lands.
        def on_anchor(idx, pose):
            viz.update(idx, pose, is_inferred=True)
        pipeline.set_anchor_callback(on_anchor)

    # Warmup the streamer (triggers torch.compile inside)
    print("[warmup] priming streamer ...", flush=True)
    t0 = time.perf_counter()
    for fr in frames[:11]:
        _ = pipeline._streamer.update(fr.to("cuda"))
    print(f"[warmup] {time.perf_counter()-t0:.1f}s\n", flush=True)
    pipeline._streamer.reset()
    pipeline._next_frame_idx = 0
    pipeline.start()

    # Feed frames at camera rate
    print(f"[sim] feeding {len(frames)} frames at {args.camera_fps} FPS ({len(frames)/args.camera_fps:.1f}s of video)\n", flush=True)
    submit_lat = []
    extrap_count_progressive = []
    inferred_count_progressive = []
    last_progress_t = time.perf_counter()
    t_start = time.perf_counter()

    for i, fr in enumerate(frames):
        target_t = t_start + i / args.camera_fps
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)
        ts = time.perf_counter() - t_start
        t_call_start = time.perf_counter()
        pose, was_extrapolated, anchor = pipeline.submit_frame(fr, ts)
        submit_lat.append((time.perf_counter() - t_call_start) * 1000.0)

        if viz is not None and pose is not None:
            # Every submit_frame call is treated as an extrapolation in the viz —
            # the actual model anchors arrive via the on_anchor callback above.
            viz.update(i, pose, is_inferred=False)

        # Progress every 2 seconds
        now2 = time.perf_counter()
        if now2 - last_progress_t > 2.0:
            s = pipeline.stats()
            print(f"  [progress] frame {i+1}/{len(frames)}  inferred={s['inferred']}  "
                  f"dropped={s['dropped']}  traj_len={s['trajectory_len']}  "
                  f"last_inf={s['last_inference_ms']:.0f}ms", flush=True)
            last_progress_t = now2

    print("\n[sim] draining ...", flush=True)
    time.sleep(2.0)
    pipeline.stop()
    t_total = time.perf_counter() - t_start

    s = pipeline.stats()
    print(f"\n[done] total wall time: {t_total:.1f}s ({len(frames)/t_total:.2f} FPS submission rate)")
    print(f"  submitted:     {s['submitted']}")
    print(f"  inferred:      {s['inferred']}  ({s['inferred']/t_total:.2f} Hz effective model rate)")
    print(f"  extrapolated:  {s['extrapolated']}")
    print(f"  dropped:       {s['dropped']}")
    print(f"  traj points:   {s['trajectory_len']}")
    print(f"  last inf ms:   {s['last_inference_ms']:.1f}")
    print(f"\n  submit_frame() latency:  mean={statistics.mean(submit_lat):.3f}ms  "
          f"p95={sorted(submit_lat)[int(0.95*len(submit_lat))]:.3f}ms  "
          f"max={max(submit_lat):.3f}ms")

    # Compare pipeline trajectory to GT
    print("\n[eval] comparing pipeline trajectory to TUM ground truth ...", flush=True)
    gt = parse_tum_groundtruth(f"{args.data}/groundtruth.txt")
    gt_per_frame = match_gt_to_frames(paths, gt)
    gt_dict = {i: g for i, g in enumerate(gt_per_frame)}
    pipe_poses = {idx: pose for idx, pose in pipeline.trajectory.all_poses()}
    common = sorted(set(gt_dict) & set(pipe_poses))
    if len(common) >= 3:
        src = np.stack([pipe_poses[i] for i in common])
        dst = np.stack([gt_dict[i] for i in common])
        scale, R, t = align_poses_sim3(src, dst)
        aligned = np.stack([apply_sim3_to_pose(p, scale, R, t) for p in src])
        rds, tds = [], []
        for i, p_a in enumerate(aligned):
            td, rd = pose_diff(dst[i], p_a)
            rds.append(rd); tds.append(td)
        print(f"  Sim(3)-aligned to GT:  scale={scale:.4f}")
        print(f"    rot RMSE: {math.sqrt(sum(r**2 for r in rds)/len(rds)):.3f}°  max: {max(rds):.3f}°")
        print(f"    trans RMSE: {math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.1f}mm  "
              f"max: {max(tds)*1000:.1f}mm")
        # Drift trend over four quarters
        q = len(rds) // 4
        if q > 0:
            quarters = [sum(tds[i*q:(i+1)*q])/q*1000 for i in range(4)]
            print(f"    trans drift Q1→Q4 (mm): {[f'{x:.1f}' for x in quarters]}")
        # Compute "true" walked distance from GT
        gt_pts = np.stack([gt_dict[i][:3, 3] for i in common])
        walked = np.linalg.norm(np.diff(gt_pts, axis=0), axis=1).sum()
        print(f"    GT walked distance: {walked:.2f} m   trajectory RMSE / walked = "
              f"{math.sqrt(sum(t**2 for t in tds)/len(tds))/walked*100:.2f}%")

    if viz is not None:
        keep_secs = args.keep_alive
        print(f"\n[viz] simulation done. Viz server staying up for {keep_secs}s "
              f"(--keep-alive). Open URL above in a browser (set up SSH tunnel if remote).")
        try:
            time.sleep(keep_secs)
        except KeyboardInterrupt:
            pass
        viz.stop()


if __name__ == "__main__":
    main()
