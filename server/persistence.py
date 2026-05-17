"""Disk persistence for recorded runs.

Each run is a self-contained JSON file in RUNS_DIR. Designed to be small enough
to ship over the wire when the /viewer loads a saved run — pointclouds are
stored quantized (int16 + scale) to keep file size down without harming visual
fidelity at room-scale geometry.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def _quantize_xyz(xyz: np.ndarray) -> dict:
    """int16 quantization: keeps ~1 mm precision at ±32 m. Lossy but fine for viz."""
    if xyz.size == 0:
        return {"q": [], "scale": 1.0, "offset": [0.0, 0.0, 0.0]}
    offset = xyz.mean(axis=0)
    centered = xyz - offset
    max_abs = float(np.abs(centered).max())
    scale = max_abs / 32700.0 if max_abs > 0 else 1.0
    q = np.round(centered / scale).clip(-32768, 32767).astype(np.int16)
    return {
        "q": q.flatten().tolist(),
        "scale": scale,
        "offset": offset.astype(float).tolist(),
    }


def _dequantize_xyz(d: dict) -> np.ndarray:
    q = np.asarray(d["q"], dtype=np.int16).reshape(-1, 3)
    return q.astype(np.float32) * float(d["scale"]) + np.asarray(d["offset"], dtype=np.float32)


class RunRecorder:
    """In-memory buffer for an active recording.

    All append_* methods are cheap (no I/O). Call ``flush_to_disk`` to save.
    """

    def __init__(self):
        self.id = uuid.uuid4().hex[:8]
        self.started_at = time.time()
        self.anchors: list[dict] = []
        self.extraps: list[dict] = []
        self.pcds: list[dict] = []

    def append_anchor(self, idx: int, t: float, pose: np.ndarray) -> None:
        self.anchors.append({"idx": int(idx), "t": float(t),
                              "pose": pose.flatten().astype(float).tolist()})

    def append_extrap(self, idx: int, t: float, pose: np.ndarray) -> None:
        # Sample every ~5th extrap to keep size bounded — visualizer can interpolate.
        if len(self.extraps) and idx - self.extraps[-1]["idx"] < 5:
            return
        self.extraps.append({"idx": int(idx), "t": float(t),
                              "pose": pose.flatten().astype(float).tolist()})

    def append_pcd(self, idx: int, xyz: np.ndarray, rgb: Optional[np.ndarray]) -> None:
        self.pcds.append({
            "idx": int(idx),
            "xyz": _quantize_xyz(xyz),
            "rgb": rgb.flatten().astype(int).tolist() if rgb is not None else None,
        })

    def flush_to_disk(self, name: Optional[str] = None) -> str:
        if name is None:
            name = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at))
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = RUNS_DIR / f"{safe}_{self.id}.json.gz"
        payload = {
            "id": self.id,
            "name": name,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "anchors": self.anchors,
            "extraps": self.extraps,
            "pcds": self.pcds,
        }
        with gzip.open(path, "wt") as f:
            json.dump(payload, f)
        return path.name


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS_DIR.glob("*.json.gz"), reverse=True):
        try:
            with gzip.open(p, "rt") as f:
                head = json.load(f)
            out.append({
                "file": p.name,
                "id": head.get("id"),
                "name": head.get("name", p.stem),
                "started_at": head.get("started_at"),
                "ended_at": head.get("ended_at"),
                "anchors": len(head.get("anchors", [])),
                "pcds": len(head.get("pcds", [])),
            })
        except Exception:
            continue
    return out


def load_run(filename: str) -> dict:
    p = RUNS_DIR / filename
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(filename)
    with gzip.open(p, "rt") as f:
        data = json.load(f)
    # Dequantize pointclouds inline so the frontend gets float arrays.
    for pcd in data.get("pcds", []):
        if isinstance(pcd.get("xyz"), dict):
            xyz = _dequantize_xyz(pcd["xyz"])
            pcd["xyz"] = xyz.flatten().tolist()
            pcd["n"] = int(xyz.shape[0])
    return data
