"""Streaming inference wrapper for LoGeR (Pi3).

Provides a stateful, frame-by-frame API on top of the existing Pi3 model. The
wrapper maintains:

  - a sliding buffer of the last ``window_size`` frames' normalized images
  - a sliding buffer of those frames' DINOv2 encoder outputs (so we re-encode
    only the newest frame on each call)
  - persistent TTT fast-weights state (``w0, w1, w2``) carried across calls
  - persistent SWA attention history carried across calls

The decoder still runs on the full window each call, so the speedup over the
batch ``model.forward(...)`` path is ``encoder_savings × (W-1)/W`` plus any
gain from ``torch.compile``. This wrapper is a foundation; true incremental
decode (one frame of work per call) is a future change requiring surgery on
the LaCT TTT and SWA attention forwards.

Accuracy note: persisting TTT/SWA state across calls is *not* bit-equivalent
to a fresh ``model.forward()`` over the buffer, because in the batch path
those states reset per-call and propagate only between internal windows.
However the streaming semantics match what the model already does between
internal windows in a multi-window forward — the state simply continues
indefinitely. Empirically (see bench/validate_streaming.py) the divergence
is small.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.amp as amp
import torch.nn as nn

from loger.models.pi3 import Pi3
from loger.utils.geometry import homogenize_points


@dataclass
class StreamingConfig:
    window_size: int = 8
    overlap_size: int = 3
    autocast_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    # If True, compile the encoder/decoder forwards with torch.compile.
    compile: bool = True
    # Reset TTT/SWA state every N updates (0 disables). Mirrors model's
    # `reset_every` kwarg used in long sequences. Only matters if persist_state.
    reset_every: int = 0
    # Persist TTT/SWA state across update() calls. The model was trained with
    # windows advancing by step=(window-overlap) frames; in streaming we
    # advance by 1 frame per call, which makes persistence drift quickly.
    # Default OFF — gives identical-to-batch poses, with encoder cache as the
    # only source of speedup. Turn on at your own risk for experiments.
    persist_state: bool = False


class StreamingLoGeR:
    """Sliding-window streaming inference for Pi3 / LoGeR.

    Usage::

        streamer = StreamingLoGeR(model, StreamingConfig(window_size=8))
        for frame in camera_stream:                  # frame: (3, H, W) on GPU
            pose = streamer.update(frame)            # (4,4) or None during warmup
            if pose is not None:
                use_pose(pose)
    """

    def __init__(self, model: Pi3, cfg: Optional[StreamingConfig] = None):
        self.model = model.eval()
        self.cfg = cfg or StreamingConfig()
        self.device = self.cfg.device

        # Compile both the encoder and decode separately. Wrapping the whole
        # `model.forward` triggers Dynamo recompiles because the windowing
        # loop produces shape variations. By splitting we keep each compiled
        # region's input shape constant: encoder always (1,3,H,W); decode
        # always (W, P, D).
        self._encoder_fn = self.model.encoder
        self._decode_fn = self.model.decode
        if self.cfg.compile:
            # mode='default': fp32-safe, no CUDA graphs (camera-head SVD would
            # break graph capture). dynamic=False -> static-shape specialization.
            self._encoder_fn = torch.compile(self._encoder_fn, mode="default",
                                              dynamic=False, fullgraph=False)
            self._decode_fn = torch.compile(self._decode_fn, mode="default",
                                             dynamic=False, fullgraph=False)

        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Drop all cached features and persistent state."""
        self._img_buf: deque[torch.Tensor] = deque(maxlen=self.cfg.window_size)
        self._feat_buf: deque[torch.Tensor] = deque(maxlen=self.cfg.window_size)
        # Persistent TTT/SWA state. Lists of (per insert-position) tensors,
        # `None` until the first decode call populates them.
        n_ttt = len(self.model.ttt_insert_after) if self.model.ttt_layers is not None else 0
        n_swa = len(self.model.attn_insert_after) if self.model.swa_layers is not None else 0
        self._w0: List[Optional[torch.Tensor]] = [None] * n_ttt
        self._w1: List[Optional[torch.Tensor]] = [None] * n_ttt
        self._w2: List[Optional[torch.Tensor]] = [None] * n_ttt
        self._swa_history: List[Optional[torch.Tensor]] = [None] * n_swa
        self._n_updates: int = 0
        self._latest_predictions: Optional[Dict[str, torch.Tensor]] = None

    @property
    def n_buffered(self) -> int:
        return len(self._feat_buf)

    @property
    def is_ready(self) -> bool:
        """True once we have a full window's worth of frames."""
        return self.n_buffered >= self.cfg.window_size

    # --------------------------------------------------------------- mutation

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the model's image_mean/std normalization."""
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, 3, H, W)
        img = img.to(self.device, non_blocking=True)
        return (img - self.model.image_mean) / self.model.image_std

    @torch.no_grad()
    def _encode_frame(self, normalized_img: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 on a single frame; return (1, num_patches, dim)."""
        with amp.autocast(self.device, dtype=self.cfg.autocast_dtype):
            feat = self._encoder_fn(normalized_img, is_training=True)
            if isinstance(feat, dict):
                feat = feat["x_norm_patchtokens"]
        return feat  # (1, P, D)

    @torch.no_grad()
    def update(self, frame: torch.Tensor) -> Optional[torch.Tensor]:
        """Feed one new frame; return camera-to-world pose for it (or None until warmup done).

        Args:
            frame: (3, H, W) or (1, 3, H, W) float tensor (values in [0,1]).

        Returns:
            (4,4) camera-to-world tensor for the newest frame on CPU, or None
            if we don't have a full window yet (during initial warmup).
        """
        norm = self._normalize(frame)             # (1, 3, H, W)
        feat = self._encode_frame(norm)           # (1, P, D)
        self._img_buf.append(norm)
        self._feat_buf.append(feat)
        self._n_updates += 1

        if not self.is_ready:
            return None

        # When state persistence is OFF (default), each call is semantically a
        # fresh single-window forward — reset transient state every time.
        if not self.cfg.persist_state:
            n_ttt = len(self._w0)
            n_swa = len(self._swa_history)
            self._w0 = [None] * n_ttt
            self._w1 = [None] * n_ttt
            self._w2 = [None] * n_ttt
            self._swa_history = [None] * n_swa
        elif (self.cfg.reset_every > 0
              and self._n_updates > self.cfg.window_size
              and self._n_updates % self.cfg.reset_every == 0):
            n_ttt = len(self._w0)
            self._w0 = [None] * n_ttt
            self._w1 = [None] * n_ttt
            self._w2 = [None] * n_ttt

        # Run the windowed decode + heads using the cached features.
        preds = self._decode_and_run_heads()
        self._latest_predictions = preds
        # Return the pose for the newest frame (last in window)
        return preds["camera_poses"][0, -1].detach().cpu()

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def _decode_and_run_heads(self) -> Dict[str, torch.Tensor]:
        """Run model.decode + the post-decode heads on the cached features."""
        m = self.model
        B = 1
        N = self.n_buffered
        # Feature tensor: each entry is (1, P, D); concat over the "frame"
        # axis to get (N, P, D). Then reshape as (B*N, P, D) for decode().
        feat = torch.cat(list(self._feat_buf), dim=0)  # (N, P, D)
        # We need H/W to compute the patch grid the decode expects.
        sample_img = self._img_buf[-1]                 # (1, 3, H, W)
        H, W = sample_img.shape[-2], sample_img.shape[-1]
        patch_h, patch_w = H // m.patch_size, W // m.patch_size

        # Build ttt/swa state dicts with our persistent buffers.
        ttt_state = None
        attn_state = None
        if m.ttt_layers is not None:
            ttt_state = {
                "ttt_op_order": m.ttt_op_order if m.ttt_op_order is not None else [],
                "insert_after": m.ttt_insert_after,
                "w0": self._w0,
                "w1": self._w1,
                "w2": self._w2,
            }
        if m.swa_layers is not None:
            attn_state = {
                "insert_after": m.attn_insert_after,
                "history": self._swa_history,
            }
        ttt_dict = None
        if ttt_state is not None or attn_state is not None:
            ttt_dict = {"ttt": ttt_state, "attn": attn_state}

        # When not persisting state, every call is the "first window" — that
        # matches model.forward's first iteration with fresh state. When
        # persisting, only the very first call is first_window.
        is_first = (not self.cfg.persist_state) or (self._n_updates == self.cfg.window_size)
        with amp.autocast(self.device, dtype=self.cfg.autocast_dtype):
            hidden, pos, ttt_output_info, _gs, _ags, _ = self._decode_fn(
                feat, N, H, W,
                ttt_dict=ttt_dict,
                window_size=self.cfg.window_size,
                overlap_size=self.cfg.overlap_size,
                is_first_window=is_first,
                turn_off_ttt=False,
                turn_off_swa=False,
            )

            # Persist updated TTT/SWA state for the next call
            if m.ttt_layers is not None and ttt_output_info is not None:
                self._w0 = ttt_output_info["w0"]
                self._w1 = ttt_output_info["w1"]
                self._w2 = ttt_output_info["w2"]
            if ttt_output_info is not None:
                self._swa_history = ttt_output_info.get("history", self._swa_history)

            # Heads
            point_hidden = m.point_decoder(hidden, xpos=pos)
            conf_hidden = m.conf_decoder(hidden, xpos=pos) if (m.use_conf and m.conf_decoder is not None) else None

            metric_hidden = None
            if m.pi3x and m.pi3x_metric and m.metric_decoder is not None:
                hw = hidden.shape[1]
                pos_hw = pos.reshape(B, N * hw, -1)
                metric_hidden = m.metric_decoder(
                    m.metric_token.repeat(B, 1, 1),
                    hidden.reshape(B, N * hw, -1),
                    xpos=pos_hw[:, 0:1], ypos=pos_hw,
                )

            camera_hidden = m.camera_decoder(hidden, xpos=pos)

        # Heads in fp32 (matches model.forward semantics)
        with torch.autocast(device_type=self.device, enabled=False):
            point_hidden = point_hidden.float()
            if m.pi3x:
                xy, z = m.point_head(point_hidden[:, m.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)
                xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
                z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
                z = torch.exp(z.clamp(max=15.0))
                local_points = torch.cat([xy * z, z], dim=-1)
            else:
                ret = m.point_head([point_hidden[:, m.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2, 1], dim=-1)
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

            if conf_hidden is not None and m.conf_head is not None:
                conf_hidden = conf_hidden.float()
                conf = m.conf_head([conf_hidden[:, m.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None

            global_camera_hidden = camera_hidden.float()
            camera_poses = m.camera_head(global_camera_hidden[:, m.patch_start_idx:],
                                          patch_h, patch_w).reshape(B, N, 4, 4)

            metric = None
            if m.pi3x and m.pi3x_metric and metric_hidden is not None and m.metric_head is not None:
                metric = m.metric_head(metric_hidden.float()).reshape(B).exp()
                camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                local_points = local_points * metric.view(B, 1, 1, 1, 1)

            points = torch.einsum('bnij, bnhwj -> bnhwi',
                                  camera_poses, homogenize_points(local_points))[..., :3]

        return {
            "camera_poses": camera_poses,
            "local_points": local_points,
            "points": points,
            "conf": conf,
            "metric": metric,
        }

    def latest_window_poses(self) -> Optional[torch.Tensor]:
        """All (N, 4, 4) poses for frames currently in the buffer (newest is last)."""
        if self._latest_predictions is None:
            return None
        return self._latest_predictions["camera_poses"][0].detach().cpu()

    def latest_window_confidence(self) -> Optional[torch.Tensor]:
        """Per-frame mean confidence for frames currently in the buffer (newest last).

        Returns a (W,) float tensor with the mean of the per-pixel sigmoid
        confidence for each frame in the window. The model's ``conf`` head emits
        raw logits with shape (B, N, H, W, 1); we apply sigmoid to map them
        into [0, 1] and then average over (H, W, 1).

        Returns ``None`` if the model has no confidence head, or if no
        prediction has been computed yet.
        """
        if self._latest_predictions is None:
            return None
        conf = self._latest_predictions.get("conf", None)
        if conf is None:
            return None
        # conf: (B, N, H, W, 1). Take batch 0, sigmoid -> [0,1], then mean over
        # spatial+channel dims to get a (N,) confidence per frame.
        conf0 = conf[0].detach()
        conf0 = torch.sigmoid(conf0.float())
        return conf0.mean(dim=(1, 2, 3)).cpu()
