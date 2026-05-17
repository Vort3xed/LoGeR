"""GlobalTrajectory: stitch streaming-window pose outputs into one world frame.

Background
----------
Each `StreamingLoGeR.update(frame)` call returns a 4×4 camera-to-world matrix
for the *latest* frame, but in the coordinate frame of the *first* frame of
the streaming buffer (which slides). Two consecutive calls produce poses in
two different coordinate frames offset by one frame's worth of motion.

This class re-anchors every call's output into a single global world frame.

How it works
------------
- The very first ready window establishes the global frame (we take its W
  per-frame poses as-is).
- Each subsequent call gives W poses for frames `[T-W+1 .. T]` in its own
  local frame. The frames `[T-W+1 .. T-1]` are also in our global trajectory
  (from previous calls). We estimate the Sim(3) transform `T_global_local`
  by aligning the local-frame poses on overlapping frames to the global ones.
- The newest frame's local pose is then mapped into global frame and stored.

Sim(3), not just SE(3), because LoGeR's metric scaling can have a tiny
multiplicative offset between consecutive calls (the metric head re-predicts
each window). Allowing scale prevents that from showing up as bogus drift.

For 1-frame-per-call streaming, the overlap is W-1 frames — plenty of
constraint for a stable Sim(3) fit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Translation-only Sim(3): solve scale*R @ src + t ≈ dst.

    src, dst: (N, 3) point clouds. Returns (scale, R(3,3), t(3,)).
    Note: For walking trajectories where the path is near-1D, R from this fit
    is under-determined. Prefer align_poses_sim3() when you have full poses.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = Vt.T @ Sgn @ U.T
    var_s = (sc ** 2).sum() / n
    scale = (S * np.diag(Sgn)).sum() / max(var_s, 1e-12)
    t = mu_d - scale * R @ mu_s
    return float(scale), R, t


def _rot_mean_so3(Rs: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Project a list of rotation matrices to their SO(3) Karcher-mean-like
    average. Cheap version: take SVD of the (optionally weighted) matrix sum.

    Rs: (N, 3, 3) rotation matrices.
    weights: optional (N,) non-negative weights; if given, each R_rel[i] is
        scaled by its weight before summing.
    """
    if weights is None:
        M = Rs.sum(axis=0)            # (3, 3)
    else:
        # Broadcast weights (N,) -> (N, 1, 1) and weight each rotation matrix.
        M = (weights[:, None, None] * Rs).sum(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] = -Vt[-1]
        R = U @ Vt
    return R


def align_poses_sim3(
    src_poses: np.ndarray,
    dst_poses: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment from POSES (not just point clouds).

    Solves for (scale, R, t) such that for each i:
        R_dst[i]      ≈  R @ R_src[i]
        t_dst[i]      ≈  scale * R @ t_src[i] + t

    Algorithm:
      1) R := SO(3)-mean of {R_dst[i] @ R_src[i]^T}     — closed form via SVD
      2) Given R, solve for (scale, t) in translations  — linear least-squares
         scale := <(t_dst - mean_dst), R @ (t_src - mean_src)> / ||t_src - mean_src||^2
         t     := mean_dst - scale * R @ mean_src

    Args:
        src_poses: (N, 4, 4) camera-to-world poses in source frame
        dst_poses: (N, 4, 4) camera-to-world poses in destination frame
    Returns:
        (scale, R(3,3), t(3,)) such that S @ P_src ≈ P_dst.
    """
    assert src_poses.shape == dst_poses.shape
    assert src_poses.ndim == 3 and src_poses.shape[1:] == (4, 4)
    N = src_poses.shape[0]
    # Per-pair relative rotations: R_rel[i] = R_dst[i] @ R_src[i]^T
    R_src = src_poses[:, :3, :3]
    R_dst = dst_poses[:, :3, :3]
    R_rels = R_dst @ R_src.transpose(0, 2, 1)        # (N, 3, 3)
    R = _rot_mean_so3(R_rels)

    t_src = src_poses[:, :3, 3]                       # (N, 3)
    t_dst = dst_poses[:, :3, 3]                       # (N, 3)
    mu_s, mu_d = t_src.mean(0), t_dst.mean(0)
    s_c = t_src - mu_s
    d_c = t_dst - mu_d
    Rs = (R @ s_c.T).T                                # (N, 3)
    num = (Rs * d_c).sum()
    den = (Rs * Rs).sum()
    scale = float(num / den) if den > 1e-12 else 1.0
    # Guard against degenerate / near-1D trajectories where the closed-form scale
    # can come out negative. Use absolute value: a negative scale would imply a
    # reflection, which Sim(3) does not allow.
    if scale <= 0:
        scale = max(abs(scale), 1e-6)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def align_poses_sim3_weighted(
    src_poses: np.ndarray,
    dst_poses: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Weighted Sim(3) alignment from POSES.

    Like :func:`align_poses_sim3`, but each per-pair contribution is scaled by
    a non-negative weight. Pairs with low weight contribute less to the
    rotation average, the translation centroid, and the scale solve.

    Args:
        src_poses: (N, 4, 4) camera-to-world poses in source frame
        dst_poses: (N, 4, 4) camera-to-world poses in destination frame
        weights:   (N,) non-negative weights. If ``None`` or constant (within
                   tolerance), reduces to the unweighted Umeyama-style solve.

    Returns:
        (scale, R(3,3), t(3,)) such that S @ P_src ≈ P_dst.
    """
    assert src_poses.shape == dst_poses.shape
    assert src_poses.ndim == 3 and src_poses.shape[1:] == (4, 4)
    N = src_poses.shape[0]

    # Sanitize weights. None or all-equal -> fall through to unweighted path.
    use_weights = False
    w: Optional[np.ndarray] = None
    if weights is not None and N > 0:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.shape == (N,), f"weights must have shape ({N},), got {w.shape}"
        # Clip negatives to 0 defensively.
        w = np.clip(w, 0.0, None)
        wsum = float(w.sum())
        if wsum > 1e-12 and not np.allclose(w, w[0]):
            # Normalize to mean=1 so weighted means stay in the same scale as
            # the unweighted code path (this keeps thresholds comparable).
            w = w * (N / wsum)
            use_weights = True

    if not use_weights:
        return align_poses_sim3(src_poses, dst_poses)

    # Per-pair relative rotations: R_rel[i] = R_dst[i] @ R_src[i]^T
    R_src = src_poses[:, :3, :3]
    R_dst = dst_poses[:, :3, :3]
    R_rels = R_dst @ R_src.transpose(0, 2, 1)        # (N, 3, 3)
    R = _rot_mean_so3(R_rels, weights=w)

    t_src = src_poses[:, :3, 3]                       # (N, 3)
    t_dst = dst_poses[:, :3, 3]                       # (N, 3)
    # Weighted means
    mu_s = np.average(t_src, axis=0, weights=w)
    mu_d = np.average(t_dst, axis=0, weights=w)
    s_c = t_src - mu_s
    d_c = t_dst - mu_d
    Rs = (R @ s_c.T).T                                # (N, 3)
    # Weighted dot/norm
    num = float(np.sum(w[:, None] * Rs * d_c))
    den = float(np.sum(w[:, None] * Rs * Rs))
    scale = float(num / den) if den > 1e-12 else 1.0
    if scale <= 0:
        scale = max(abs(scale), 1e-6)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_sim3_to_pose(pose: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply Sim(3) (scale, R, t) to a 4x4 camera-to-world pose.

    Under Sim(3) reframe: t_new = scale*R @ t_old + t_offset
                          R_new = R @ R_old
    """
    out = pose.copy()
    out[:3, 3] = scale * (R @ pose[:3, 3]) + t
    out[:3, :3] = R @ pose[:3, :3]
    return out


def pose_to_qt(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose a 4x4 SE(3) pose into (quaternion (4,), translation (3,))."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    # Quaternion from rotation matrix (Shepperd's method)
    tr = R.trace()
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz]), t.copy()


def qt_to_pose(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) from quaternion (w,x,y,z) and translation (3,)."""
    qw, qx, qy, qz = q
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(4)
    qw, qx, qy, qz = q / norm
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (w,x,y,z)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(d)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * u
    s0 = np.cos(theta) - d * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


@dataclass
class _Frame:
    idx: int                # global frame index (0,1,2,...)
    timestamp: float        # arrival time (seconds)
    pose: np.ndarray        # (4,4) in global world frame


class GlobalTrajectory:
    """Thread-safe accumulator of camera-to-world poses in one global frame.

    Public API
    ----------
    - ingest_window(window_poses, newest_idx, timestamp): take a (W,4,4)
      window from a streaming call, with frame `newest_idx` being the latest.
      Performs Sim(3) stitching and appends the result.
    - latest(): -> (idx, pose) or None.
    - get_pose(idx): -> 4x4 pose at that frame, or None.
    - extrapolate(now_t): -> (4,4) predicted pose at time `now_t` using a
      simple constant-velocity model on the most recent N poses.
    """

    def __init__(self, smoothing_alpha: float = 0.5):
        """smoothing_alpha in [0,1]: when re-stitching an already-known frame,
        we blend old_pose * (1-α) + new_pose * α. α=1.0 = always overwrite."""
        self._lock = threading.Lock()
        self._frames: Dict[int, _Frame] = {}
        self._order: List[int] = []   # global indices in insertion order
        self._smoothing_alpha = float(smoothing_alpha)

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def latest(self) -> Optional[Tuple[int, np.ndarray]]:
        with self._lock:
            if not self._order:
                return None
            i = self._order[-1]
            return i, self._frames[i].pose.copy()

    def get_pose(self, idx: int) -> Optional[np.ndarray]:
        with self._lock:
            f = self._frames.get(idx)
            return f.pose.copy() if f is not None else None

    def all_poses(self) -> List[Tuple[int, np.ndarray]]:
        with self._lock:
            return [(i, self._frames[i].pose.copy()) for i in self._order]

    # ------------------------------------------------------------ stitching

    def ingest_window(self, window_poses: np.ndarray, newest_idx: int,
                       timestamp: float) -> Tuple[int, np.ndarray]:
        """Stitch a fresh (W,4,4) window from a streaming call.

        Args:
            window_poses: (W, 4, 4) local-frame poses, oldest first, newest last.
            newest_idx:   global frame index of the LAST pose in `window_poses`.
            timestamp:    arrival time of this newest frame (seconds).

        Returns:
            (newest_idx, world_pose) — the newest frame's pose in global frame.
        """
        W = window_poses.shape[0]
        assert window_poses.shape == (W, 4, 4)
        oldest_idx = newest_idx - (W - 1)

        with self._lock:
            if not self._order:
                # First call ever: this window IS the global frame.
                for k in range(W):
                    gi = oldest_idx + k
                    self._frames[gi] = _Frame(gi, timestamp, window_poses[k].copy())
                    self._order.append(gi)
                return newest_idx, window_poses[-1].copy()

            # Find overlapping frames: those in [oldest_idx .. newest_idx-1]
            # that already exist in self._frames. Use the FULL POSES (not just
            # translations) for Sim(3) alignment — translation-only Umeyama is
            # ill-conditioned on near-linear walking paths.
            overlap_local_poses = []
            overlap_global_poses = []
            for k in range(W - 1):
                gi = oldest_idx + k
                if gi in self._frames:
                    overlap_local_poses.append(window_poses[k])
                    overlap_global_poses.append(self._frames[gi].pose)
            if len(overlap_local_poses) >= 1:
                src = np.stack(overlap_local_poses)
                dst = np.stack(overlap_global_poses)
                scale, R, t = align_poses_sim3(src, dst)
            else:
                scale, R, t = 1.0, np.eye(3), np.zeros(3)

            # ANCHOR rule: a frame is established the first time it appears as
            # the newest in a window, and is then immutable. Subsequent calls
            # only add the new frame(s), anchored to the existing global trajectory
            # via Sim(3) on the overlap. This prevents cumulative drift from
            # repeated re-alignment of the same frame.
            world_pose_newest = None
            for k in range(W):
                gi = oldest_idx + k
                if gi in self._frames:
                    # Frame already anchored. Skip overwriting unless smoothing < 1.
                    if self._smoothing_alpha < 1.0 and k != W - 1:
                        # Optional gentle correction (mostly for translation drift)
                        α = self._smoothing_alpha
                        old = self._frames[gi].pose
                        world_pose = apply_sim3_to_pose(window_poses[k], scale, R, t)
                        new_t = (1 - α) * old[:3, 3] + α * world_pose[:3, 3]
                        q_old, _ = pose_to_qt(old)
                        q_new, _ = pose_to_qt(world_pose)
                        q_blend = slerp(q_old, q_new, α)
                        self._frames[gi].pose = qt_to_pose(q_blend, new_t)
                    if k == W - 1:
                        world_pose_newest = self._frames[gi].pose.copy()
                else:
                    world_pose = apply_sim3_to_pose(window_poses[k], scale, R, t)
                    self._frames[gi] = _Frame(gi, timestamp, world_pose)
                    self._order.append(gi)
                    if k == W - 1:
                        world_pose_newest = world_pose.copy()

            return newest_idx, world_pose_newest  # type: ignore[return-value]

    def ingest_window_sparse(self, window_poses: np.ndarray,
                               cam_indices: list, timestamp: float,
                               confidence: Optional[np.ndarray] = None,
                               ) -> Tuple[int, np.ndarray]:
        """Like ingest_window but the W poses correspond to non-consecutive
        camera frame indices `cam_indices` (oldest first, newest last).

        Use this when the inference worker drops frames — the streamer's buffer
        contains a sparse sample of camera frames, not a consecutive run.

        Args:
            window_poses: (W, 4, 4) per-frame local poses for this window.
            cam_indices:  global frame indices for each window slot (ascending).
            timestamp:    arrival time of the newest frame.
            confidence:   optional (W,) array of per-frame model confidences.
                          When provided, overlap frames are weighted by their
                          confidence during Sim(3) alignment, so low-confidence
                          frames pull less on the global stitch. ``None`` means
                          unweighted (legacy behavior).
        """
        W = window_poses.shape[0]
        assert window_poses.shape == (W, 4, 4)
        assert len(cam_indices) == W
        assert cam_indices == sorted(cam_indices), "cam_indices must be ascending"
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
            assert confidence.shape == (W,), \
                f"confidence must have shape ({W},), got {confidence.shape}"
        newest_idx = cam_indices[-1]

        with self._lock:
            if not self._order:
                for k, gi in enumerate(cam_indices):
                    self._frames[gi] = _Frame(gi, timestamp, window_poses[k].copy())
                    self._order.append(gi)
                self._order.sort()
                return newest_idx, window_poses[-1].copy()

            # Find overlap on the cam_indices basis (not arithmetic)
            overlap_local_poses = []
            overlap_global_poses = []
            overlap_weights: List[float] = []
            for k in range(W - 1):
                gi = cam_indices[k]
                if gi in self._frames:
                    overlap_local_poses.append(window_poses[k])
                    overlap_global_poses.append(self._frames[gi].pose)
                    if confidence is not None:
                        overlap_weights.append(float(confidence[k]))
            if len(overlap_local_poses) >= 1:
                src = np.stack(overlap_local_poses)
                dst = np.stack(overlap_global_poses)
                if confidence is not None and len(overlap_weights) == len(overlap_local_poses):
                    w_arr = np.asarray(overlap_weights, dtype=np.float64)
                    scale, R, t = align_poses_sim3_weighted(src, dst, w_arr)
                else:
                    scale, R, t = align_poses_sim3(src, dst)
            else:
                scale, R, t = 1.0, np.eye(3), np.zeros(3)

            world_pose_newest = None
            for k, gi in enumerate(cam_indices):
                if gi in self._frames:
                    if self._smoothing_alpha < 1.0 and k != W - 1:
                        α = self._smoothing_alpha
                        old = self._frames[gi].pose
                        world_pose = apply_sim3_to_pose(window_poses[k], scale, R, t)
                        new_t = (1 - α) * old[:3, 3] + α * world_pose[:3, 3]
                        q_old, _ = pose_to_qt(old)
                        q_new, _ = pose_to_qt(world_pose)
                        q_blend = slerp(q_old, q_new, α)
                        self._frames[gi].pose = qt_to_pose(q_blend, new_t)
                    if k == W - 1:
                        world_pose_newest = self._frames[gi].pose.copy()
                else:
                    world_pose = apply_sim3_to_pose(window_poses[k], scale, R, t)
                    self._frames[gi] = _Frame(gi, timestamp, world_pose)
                    self._order.append(gi)
                    if k == W - 1:
                        world_pose_newest = world_pose.copy()
            self._order.sort()
            return newest_idx, world_pose_newest  # type: ignore[return-value]

    # ----------------------------------------------------------- prediction

    def extrapolate(self, now_t: float, lookback: int = 2) -> Optional[np.ndarray]:
        """Predict pose at time `now_t` from recent fused poses.

        Constant-velocity from the LAST TWO anchors (translation) + slerp
        extrapolation on the last two quaternions (rotation). Empirically this
        beats averaging over more segments for walking-style motion, because
        the most recent velocity tracks accelerations / direction changes better
        than a smoothed estimate (which lags reality).
        """
        with self._lock:
            if len(self._order) < 2:
                return None
            recent_idx = self._order[-lookback:] if len(self._order) >= lookback else self._order
            frames = [self._frames[i] for i in recent_idx]

        if len(frames) < 2:
            return None

        last = frames[-1]
        prev = frames[-2]
        dt_last = last.timestamp - prev.timestamp
        dt_extrap = now_t - last.timestamp

        if dt_last <= 0:
            return last.pose.copy()

        # Translation: linear extrapolation along last-segment velocity
        v = (last.pose[:3, 3] - prev.pose[:3, 3]) / dt_last
        t_pred = last.pose[:3, 3] + v * dt_extrap

        # Rotation: slerp from prev to last, extrapolated to u = 1 + dt/dt_last
        q_prev, _ = pose_to_qt(prev.pose)
        q_last, _ = pose_to_qt(last.pose)
        u = 1.0 + dt_extrap / dt_last
        q_pred = slerp(q_prev, q_last, u)

        return qt_to_pose(q_pred, t_pred)
