"""Measure how good the extrapolation is.

For each camera frame T at time t_T, the pipeline returns one of:
  (a) an INFERRED pose, when a model inference just completed for frame T
  (b) an EXTRAPOLATED pose, computed from constant-velocity on recent anchors

We have GT (TUM Vicon) for every frame. We compare:
  - extrap_err[T]:  ||extrapolated_pose[T] - GT[T]||  (Sim(3)-aligned)
  - holdlast_err[T]: ||last_inferred_pose - GT[T]||   (the "no extrapolation" baseline)
  - inferred_err[T]: ||inferred_pose[T] - GT[T]||      (for anchor frames)

If extrap_err << holdlast_err, extrapolation is adding real value.
If extrap_err ≈ holdlast_err, extrapolation is fluff and we should just hold-last.
If extrap_err > holdlast_err, extrapolation is actively harmful (overshooting).

We also break down error by *time-since-last-anchor* — extrapolation should be
near-zero error for the first 30 ms and grow as the gap widens.
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

sys.path.insert(0, str(REPO / "bench"))
from stress_and_viz import (load_model, build_frames, parse_tum_groundtruth,
                              match_gt_to_frames, pose_diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/rgbd_dataset_freiburg3_long_office_household")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--camera-fps", type=int, default=30)
    args = ap.parse_args()

    print(f"[setup] loading model + {args.frames} frames ...", flush=True)
    model = load_model()
    frames, paths, (TW, TH) = build_frames(f"{args.data}/rgb", args.frames)
    print(f"[data] {len(frames)} frames @ {TW}x{TH}\n", flush=True)

    pipeline = RealtimePipeline(model, PipelineConfig(
        window_size=8, overlap_size=3, compile=True, inference_queue_size=1))

    # Warmup
    print("[warmup] priming streamer ...", flush=True)
    t0 = time.perf_counter()
    for fr in frames[:11]:
        _ = pipeline._streamer.update(fr.to("cuda"))
    print(f"[warmup] {time.perf_counter()-t0:.1f}s\n", flush=True)
    pipeline._streamer.reset()
    pipeline._next_frame_idx = 0
    pipeline.start()

    # Capture per-frame: (frame_idx, pose, was_extrapolated, timestamp,
    #                     anchor_idx_at_time_of_call)
    records = []
    t_start = time.perf_counter()
    for i, fr in enumerate(frames):
        target_t = t_start + i / args.camera_fps
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)
        ts = time.perf_counter() - t_start
        pose, was_extrap, anchor = pipeline.submit_frame(fr, ts)
        records.append({
            "frame_idx": i,
            "timestamp": ts,
            "pose": pose.copy() if pose is not None else None,
            "extrapolated": bool(was_extrap),
            "anchor_idx_at_call": anchor,
        })

    print("[sim] draining ...", flush=True)
    time.sleep(2.0)
    pipeline.stop()
    t_total = time.perf_counter() - t_start
    print(f"[sim] done ({t_total:.1f}s)\n", flush=True)

    # ===== Identify anchored (inferred-via-model) frames =====
    # The trajectory holds only the anchor frames. Everything else in `records`
    # is an extrapolation.
    anchored_frames = {idx for idx, _ in pipeline.trajectory.all_poses()}
    print(f"[data] {len(anchored_frames)} anchor frames, "
          f"{len(records) - len(anchored_frames)} extrapolated calls")

    # ===== GT =====
    gt = parse_tum_groundtruth(f"{args.data}/groundtruth.txt")
    gt_per_frame = match_gt_to_frames(paths, gt)
    gt_dict = {i: g for i, g in enumerate(gt_per_frame)}

    # ===== Sim(3) align pipeline frame -> GT using ANCHOR frames only =====
    # (Anchors are the model's authoritative output; extrapolations are noisy
    # predictions and shouldn't drive the alignment.)
    pipe_poses = {idx: p for idx, p in pipeline.trajectory.all_poses()}
    common_anchors = sorted(anchored_frames & set(gt_dict.keys()))
    if len(common_anchors) < 3:
        print("Not enough anchors to align. Bail.")
        return

    src = np.stack([pipe_poses[i] for i in common_anchors])
    dst = np.stack([gt_dict[i] for i in common_anchors])
    scale, R, t = align_poses_sim3(src, dst)
    print(f"\n[align] anchors->GT Sim(3): scale={scale:.4f}\n", flush=True)

    # ===== For each frame, compute error of extrap vs GT and hold-last vs GT =====
    # Hold-last baseline: at time of frame T, what was the most recent ANCHOR
    # pose? Apply Sim(3), compare to GT.
    by_time_bucket: dict = {}  # bucket_idx (frames-since-anchor) -> list of (extrap_err, holdlast_err, t_err_extrap, t_err_holdlast)

    last_anchor_idx = -1
    extrap_t_errs = []
    holdlast_t_errs = []
    extrap_r_errs = []
    holdlast_r_errs = []
    inferred_t_errs = []
    inferred_r_errs = []

    for rec in records:
        i = rec["frame_idx"]
        if rec["pose"] is None:
            continue
        if i not in gt_dict:
            continue
        # Bring extrap pose into GT frame
        extrap_aligned = apply_sim3_to_pose(rec["pose"], scale, R, t)
        gt_pose = gt_dict[i]
        td_extrap, rd_extrap = pose_diff(gt_pose, extrap_aligned)

        # Identify last anchor (the most recent inferred frame, BEFORE or AT frame i)
        anchor_idx = rec["anchor_idx_at_call"]
        if anchor_idx >= 0 and anchor_idx in pipe_poses:
            last_anchor_pose_aligned = apply_sim3_to_pose(pipe_poses[anchor_idx], scale, R, t)
            td_hl, rd_hl = pose_diff(gt_pose, last_anchor_pose_aligned)
        else:
            td_hl, rd_hl = None, None

        # Categorize: was this an inferred frame (anchor) or extrapolated?
        if i in anchored_frames:
            inferred_t_errs.append(td_extrap)
            inferred_r_errs.append(rd_extrap)
        else:
            extrap_t_errs.append(td_extrap)
            extrap_r_errs.append(rd_extrap)
            if td_hl is not None:
                holdlast_t_errs.append(td_hl)
                holdlast_r_errs.append(rd_hl)

                # Finer buckets — at 3 Hz inference / 30 Hz camera, gaps cluster around 10-25
                gap = i - anchor_idx
                if gap <= 5:    bucket = "1-5"
                elif gap <= 10: bucket = "6-10"
                elif gap <= 15: bucket = "11-15"
                elif gap <= 20: bucket = "16-20"
                elif gap <= 30: bucket = "21-30"
                else:           bucket = "31+"
                by_time_bucket.setdefault(bucket, []).append((td_extrap, td_hl, rd_extrap, rd_hl, gap))

    # ===== Report =====
    def stats(xs, mult=1000.0):
        if not xs:
            return "(no data)"
        return (f"mean={sum(xs)/len(xs)*mult:.2f}  "
                f"median={sorted(xs)[len(xs)//2]*mult:.2f}  "
                f"p95={sorted(xs)[int(0.95*len(xs))]*mult:.2f}  "
                f"max={max(xs)*mult:.2f}")

    print("=" * 78)
    print("ABSOLUTE ERROR vs GROUND TRUTH")
    print("=" * 78)
    print(f"  INFERRED (model output, anchored frames):  n={len(inferred_t_errs)}")
    print(f"    trans error (mm):  {stats(inferred_t_errs)}")
    print(f"    rot   error (deg): {stats(inferred_r_errs, mult=1.0)}")
    print()
    print(f"  EXTRAPOLATED (pipeline prediction for non-anchor frames):  n={len(extrap_t_errs)}")
    print(f"    trans error (mm):  {stats(extrap_t_errs)}")
    print(f"    rot   error (deg): {stats(extrap_r_errs, mult=1.0)}")
    print()
    print(f"  HOLD-LAST (baseline: just keep showing last anchor):  n={len(holdlast_t_errs)}")
    print(f"    trans error (mm):  {stats(holdlast_t_errs)}")
    print(f"    rot   error (deg): {stats(holdlast_r_errs, mult=1.0)}")
    print()

    if holdlast_t_errs and extrap_t_errs:
        gain_pct = (sum(holdlast_t_errs)/len(holdlast_t_errs) -
                    sum(extrap_t_errs)/len(extrap_t_errs)) / (sum(holdlast_t_errs)/len(holdlast_t_errs)) * 100
        print(f"  → Extrapolation reduces mean trans error by {gain_pct:+.1f}% vs hold-last")

    print()
    print("=" * 78)
    print("BY TIME-SINCE-ANCHOR (gap = frame_idx - last_anchor_idx)")
    print("=" * 78)
    print(f"{'bucket':>7}  {'n':>5}  {'extrap mm':>14}  {'hold-last mm':>14}  {'reduce%':>8}  {'extrap°':>9}  {'hold-last°':>11}")
    for bucket in ["1-5", "6-10", "11-15", "16-20", "21-30", "31+"]:
        if bucket not in by_time_bucket:
            continue
        rows = by_time_bucket[bucket]
        et = [r[0] for r in rows]
        ht = [r[1] for r in rows]
        er = [r[2] for r in rows]
        hr = [r[3] for r in rows]
        et_mean = sum(et)/len(et)*1000
        ht_mean = sum(ht)/len(ht)*1000
        reduction = (ht_mean - et_mean) / ht_mean * 100 if ht_mean > 0 else 0
        print(f"  {bucket:>5}    {len(rows):>5}  "
              f"{et_mean:11.2f}     {ht_mean:11.2f}     "
              f"{reduction:+6.1f}%   "
              f"{sum(er)/len(er):7.2f}      "
              f"{sum(hr)/len(hr):8.2f}")


if __name__ == "__main__":
    main()
