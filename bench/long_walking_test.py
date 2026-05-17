"""Long real-walking test on TUM freiburg3_walking_xyz (859 frames, ~30 sec walk).

Three things being measured:
  1. SYNCHRONOUS drift: stream all N frames through StreamingLoGeR + GlobalTrajectory
     synchronously (no dropping). Compare to batch model.forward(all N) reference.
     This tells us the *stitching* quality on a long sequence.
  2. PIPELINE simulation: feed frames at 30 FPS camera rate through RealtimePipeline.
     Measure submit_frame latency, fraction extrapolated, and trajectory accuracy.
     This is the realistic real-time scenario.
  3. GROUND-TRUTH comparison: TUM provides ground-truth poses (Vicon). Compare both
     model output and our pipeline against actual physics.
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
from loger.trajectory import GlobalTrajectory, align_poses_sim3, apply_sim3_to_pose
from loger.realtime_pipeline import RealtimePipeline, PipelineConfig


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


def build_frames_from_dir(image_dir, n, pixel_limit=255_000):
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
             for p in paths],
            paths,
            (TW, TH))


def parse_tum_groundtruth(path):
    """Return list of (timestamp, 4x4 pose) tuples from TUM groundtruth.txt."""
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
            # Convert quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
            ])
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
            out.append((ts, T))
    return out


def match_gt_to_frames(rgb_paths, gt):
    """For each rgb timestamp, pick the closest gt entry. Returns list of (rgb_path, gt_pose)."""
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
    rd = math.degrees(math.acos(c))
    return td, rd


def report_drift(name, ref_poses, test_poses, indices):
    """Sim(3)-align test→ref over `indices`, report per-frame residuals + trend."""
    src = np.stack([test_poses[i] for i in indices])
    dst = np.stack([ref_poses[i] for i in indices])
    scale, R, t = align_poses_sim3(src, dst)
    aligned = np.stack([apply_sim3_to_pose(p, scale, R, t) for p in src])
    rds, tds = [], []
    for i, p_a in enumerate(aligned):
        td, rd = pose_diff(dst[i], p_a)
        rds.append(rd); tds.append(td)
    print(f"  [{name}]  scale={scale:.4f}    "
          f"rot RMSE={math.sqrt(sum(r**2 for r in rds)/len(rds)):.3f}°  max={max(rds):.3f}°    "
          f"trans RMSE={math.sqrt(sum(t**2 for t in tds)/len(tds))*1000:.1f}mm  "
          f"max={max(tds)*1000:.1f}mm")
    # Linear drift trend
    quarter = len(rds) // 4
    if quarter > 0:
        early = sum(tds[:quarter]) / quarter * 1000
        late = sum(tds[-quarter:]) / quarter * 1000
        print(f"     trans drift  early-q mean={early:.1f}mm  late-q mean={late:.1f}mm  Δ={late-early:+.1f}mm")
    return scale, math.sqrt(sum(r**2 for r in rds)/len(rds))


def main():
    DATA = "/tmp/rgbd_dataset_freiburg3_long_office_household"
    N_FRAMES = 300  # 300 frames @ ~30 FPS = ~10 sec of real office walking
    # The full sequence has 2585 frames (~87 sec); 300 is tractable for batch reference

    print(f"[setup] loading model + {N_FRAMES} TUM walking frames ...", flush=True)
    model = load_model()
    frames, paths, (TW, TH) = build_frames_from_dir(f"{DATA}/rgb", N_FRAMES)
    print(f"[data] {len(frames)} frames @ {TW}x{TH}\n", flush=True)

    # ===== Phase A: synchronous drift test =====
    print("=" * 78)
    print("Phase A: synchronous streaming + GlobalTrajectory vs batch reference")
    print("=" * 78)
    # batch reference
    print("[ref] batch forward (all frames in one call, W=8 OV=3) ...", flush=True)
    x = torch.stack(frames).unsqueeze(0).to("cuda")
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, window_size=8, overlap_size=3)
    ref_poses = {i: out["camera_poses"][0, i].float().cpu().numpy() for i in range(len(frames))}
    del out, x
    torch.cuda.empty_cache()
    print("[ref] done\n", flush=True)

    # streaming + sync stitching
    streamer = StreamingLoGeR(model, StreamingConfig(window_size=8, overlap_size=3,
                                                       compile=False, persist_state=False))
    traj_sync = GlobalTrajectory(smoothing_alpha=1.0)
    for i, fr in enumerate(frames):
        _ = streamer.update(fr.to("cuda"))
        if streamer.is_ready:
            win = streamer.latest_window_poses()
            if win is not None:
                traj_sync.ingest_window(win.numpy(), newest_idx=i, timestamp=float(i)/30.0)
    common = sorted(set(ref_poses) & set(idx for idx, _ in traj_sync.all_poses()))
    test_poses = {idx: pose for idx, pose in traj_sync.all_poses()}
    report_drift("sync-stitched", ref_poses, test_poses, common)
    del streamer, traj_sync
    torch.cuda.empty_cache()

    # ===== Phase B: concurrent pipeline at 30 FPS =====
    print("\n" + "=" * 78)
    print("Phase B: RealtimePipeline at 30 FPS camera rate")
    print("=" * 78)
    pipeline = RealtimePipeline(model, PipelineConfig(window_size=8, overlap_size=3,
                                                       compile=True, inference_queue_size=1))
    # Warmup
    print("[warmup] priming streamer (triggers torch.compile) ...", flush=True)
    t_warm = time.perf_counter()
    for fr in frames[:11]:
        _ = pipeline._streamer.update(fr.to("cuda"))
    print(f"[warmup] done in {time.perf_counter()-t_warm:.1f}s\n", flush=True)
    pipeline._streamer.reset()
    pipeline._next_frame_idx = 0
    pipeline.start()

    CAMERA_FPS = 30
    submit_latencies_ms = []
    extrap_flags = []
    t_start = time.perf_counter()
    for i, fr in enumerate(frames):
        target_t = t_start + i / CAMERA_FPS
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)
        timestamp = time.perf_counter() - t_start
        t0 = time.perf_counter()
        pose, extrap, anchor = pipeline.submit_frame(fr, timestamp)
        submit_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        extrap_flags.append(extrap)

    print("[sim] draining ...", flush=True)
    time.sleep(1.0)
    pipeline.stop()
    print(f"[sim] done\n", flush=True)

    s = pipeline.stats()
    print(f"  frames submitted:   {s['submitted']}")
    print(f"  inferred frames:    {s['inferred']}  (model rate {s['inferred']/(len(frames)/CAMERA_FPS):.2f} Hz)")
    print(f"  extrapolated calls: {s['extrapolated']}")
    print(f"  dropped for inf:    {s['dropped']}")
    print(f"  trajectory length:  {s['trajectory_len']}")
    print(f"  submit_frame() latency: mean={statistics.mean(submit_latencies_ms):.2f}ms  "
          f"p95={sorted(submit_latencies_ms)[int(0.95*len(submit_latencies_ms))]:.2f}ms  "
          f"max={max(submit_latencies_ms):.2f}ms")

    pipe_poses = {idx: pose for idx, pose in pipeline.trajectory.all_poses()}
    common_pipe = sorted(set(ref_poses) & set(pipe_poses))
    if len(common_pipe) >= 3:
        report_drift("pipeline-stitched", ref_poses, pipe_poses, common_pipe)

    # ===== Phase C: ground truth comparison =====
    print("\n" + "=" * 78)
    print("Phase C: comparison against TUM ground truth (Vicon)")
    print("=" * 78)
    gt = parse_tum_groundtruth(f"{DATA}/groundtruth.txt")
    gt_per_frame = match_gt_to_frames(paths, gt)
    gt_dict = {i: g for i, g in enumerate(gt_per_frame)}
    print(f"[gt] {len(gt)} ground-truth entries, matched to {len(gt_per_frame)} rgb frames")
    print(f"[gt] sample: rgb_ts={Path(paths[0]).stem}  gt_pose[0]=\n{gt_dict[0]}\n")

    # Compare ref (batch model.forward) to GT
    print("Reference (batch model.forward) vs GT:")
    report_drift("ref vs GT", gt_dict, ref_poses, sorted(ref_poses.keys()))
    # Compare pipeline-stitched to GT
    if pipe_poses:
        print("Pipeline-stitched vs GT:")
        report_drift("pipeline vs GT", gt_dict, pipe_poses, sorted(pipe_poses.keys()))


if __name__ == "__main__":
    main()
