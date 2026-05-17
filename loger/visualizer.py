"""Live 3D trajectory visualizer for the real-time pipeline.

Spins up a Viser HTTP server and exposes:
  - growing trajectory as a 3D polyline
  - current camera position (sphere)
  - current camera orientation (frustum)
  - on-screen counters: total frames received, frames inferred, FPS

Usage::

    from loger.visualizer import TrajectoryVisualizer
    viz = TrajectoryVisualizer(port=8080)
    viz.start()
    # then, in your pipeline loop, on each new fused pose:
    viz.update(idx, pose_4x4, is_extrapolated=False)
    # ...
    viz.stop()

The viz is decoupled from the pipeline — you call ``update()`` whenever you have
a new pose, whether from a model inference or an extrapolation. The viz
distinguishes "precise" (inferred) from "extrapolated" via color.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Dict, Tuple

import numpy as np


# Viser is heavy; defer import to start() so import-time stays cheap and
# headless environments can still import this module for type-checking.
def _import_viser():
    import viser  # noqa: F401
    return viser


class TrajectoryVisualizer:
    """Web-based live viz for streaming camera poses."""

    def __init__(self, port: int = 8080, trail_length: int = 5000,
                 frustum_scale: float = 0.05):
        self.port = port
        self.trail_length = trail_length
        self.frustum_scale = frustum_scale

        self._server = None
        self._lock = threading.Lock()

        # Trajectory state (numpy, thread-safe via lock)
        self._traj_positions: deque[np.ndarray] = deque(maxlen=trail_length)
        self._traj_inferred_mask: deque[bool] = deque(maxlen=trail_length)
        self._latest_pose: Optional[np.ndarray] = None
        self._latest_is_inferred: bool = False
        self._latest_idx: int = -1

        # Stats
        self._n_inferred = 0
        self._n_extrapolated = 0
        self._t_started: Optional[float] = None
        self._t_last_inferred: Optional[float] = None

        # Scene handles populated on start()
        self._traj_handle = None
        self._extrap_handle = None
        self._frustum_handle = None
        self._sphere_handle = None
        self._counters_handle = None

        self._render_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- start

    def start(self) -> str:
        viser = _import_viser()
        self._server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self._t_started = time.perf_counter()

        # Coordinate frame so the viewer knows orientation
        self._server.scene.add_frame("/world", show_axes=True, axes_length=0.5, axes_radius=0.005)
        # Initial empty trajectory line (precise)
        self._traj_handle = self._server.scene.add_spline_catmull_rom(
            "/world/trajectory",
            positions=np.zeros((2, 3), dtype=np.float32),
            color=(0.1, 0.8, 0.3),
            line_width=2.0,
        )
        # Extrapolated trail (lighter color)
        self._extrap_handle = self._server.scene.add_spline_catmull_rom(
            "/world/extrapolated",
            positions=np.zeros((2, 3), dtype=np.float32),
            color=(0.9, 0.7, 0.2),
            line_width=1.0,
        )
        # Current-pose marker: a small sphere
        self._sphere_handle = self._server.scene.add_icosphere(
            "/world/current",
            radius=0.015,
            color=(1.0, 0.3, 0.3),
        )
        # Camera frustum showing the current orientation
        self._frustum_handle = self._server.scene.add_camera_frustum(
            "/world/frustum",
            fov=1.0,
            aspect=1.33,
            scale=self.frustum_scale,
            color=(0.3, 0.7, 1.0),
        )
        self._counters_handle = self._server.gui.add_markdown("**Waiting for frames...**")

        self._render_thread = threading.Thread(target=self._render_loop, daemon=True,
                                                name="viser-render")
        self._render_thread.start()
        url = f"http://0.0.0.0:{self.port}"
        print(f"[viz] Viser server started at {url}", flush=True)
        return url

    def stop(self) -> None:
        self._stop.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass

    # ------------------------------------------------------------- update

    def update(self, frame_idx: int, pose: np.ndarray, is_inferred: bool) -> None:
        """Push a new pose. ``pose`` is 4x4 camera-to-world. ``is_inferred``
        means "this came from a real model inference" (vs. extrapolated)."""
        assert pose.shape == (4, 4)
        with self._lock:
            pos = pose[:3, 3].astype(np.float32)
            self._traj_positions.append(pos)
            self._traj_inferred_mask.append(bool(is_inferred))
            self._latest_pose = pose.copy()
            self._latest_is_inferred = bool(is_inferred)
            self._latest_idx = frame_idx
            if is_inferred:
                self._n_inferred += 1
                self._t_last_inferred = time.perf_counter()
            else:
                self._n_extrapolated += 1

    # ----------------------------------------------------------- internals

    def _render_loop(self):
        """Pull state under lock, push to Viser. ~30 FPS render."""
        target_dt = 1.0 / 30.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._render_once()
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def _render_once(self):
        with self._lock:
            if self._latest_pose is None:
                return
            # Snapshot copies
            positions = list(self._traj_positions)
            inferred_mask = list(self._traj_inferred_mask)
            latest_pose = self._latest_pose.copy()
            latest_idx = self._latest_idx
            n_inferred = self._n_inferred
            n_extrapolated = self._n_extrapolated
            t_start = self._t_started

        # Split positions into inferred-only and full trails
        inferred_positions = [p for p, ok in zip(positions, inferred_mask) if ok]
        all_positions = positions

        # Trajectory line: only precise (inferred) anchors
        if len(inferred_positions) >= 2:
            self._traj_handle.positions = np.stack(inferred_positions)
        # Extrapolated trail: the full trail (includes extrapolations) lighter color
        if len(all_positions) >= 2:
            self._extrap_handle.positions = np.stack(all_positions)

        # Current marker + frustum
        self._sphere_handle.position = latest_pose[:3, 3].astype(np.float32)
        # Frustum needs wxyz quaternion + position
        from loger.trajectory import pose_to_qt
        q, t = pose_to_qt(latest_pose)
        self._frustum_handle.position = t.astype(np.float32)
        self._frustum_handle.wxyz = q.astype(np.float32)
        # Frustum color depends on whether latest is inferred or extrapolated
        if self._latest_is_inferred:
            self._frustum_handle.color = (0.1, 0.9, 0.3)
        else:
            self._frustum_handle.color = (0.9, 0.7, 0.2)

        # Counters
        elapsed = time.perf_counter() - (t_start or time.perf_counter())
        total = n_inferred + n_extrapolated
        infer_rate = n_inferred / elapsed if elapsed > 0 else 0
        total_rate = total / elapsed if elapsed > 0 else 0
        md = (
            f"### Real-time tracking\n"
            f"- Frame index: **{latest_idx}**\n"
            f"- Inferred anchors: **{n_inferred}**  ({infer_rate:.1f} Hz)\n"
            f"- Total poses (incl. extrapolations): **{total}**  ({total_rate:.1f} Hz)\n"
            f"- Elapsed: {elapsed:.1f} s\n"
            f"- Last update: **{'INFERRED' if self._latest_is_inferred else 'extrapolated'}**\n"
        )
        try:
            self._counters_handle.content = md
        except Exception:
            pass
