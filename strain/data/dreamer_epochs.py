"""
Memmap-backed DREAMER epoch tensors produced by ``scripts/export_dreamer_epochs.py``.

Layout under ``data/processed/dreamer/`` (gitignored with ``data/``):

- ``X.npy`` — memory-mappable array ``(n_epochs, 14, chunk_size)`` float32
- ``subject_id.npy``, ``trial_id.npy``, ``start_sample.npy`` — int32
- ``valence.npy``, ``arousal.npy``, ``dominance.npy`` — float32 (1–5 Likert)
- ``manifest.json`` — version, paths, shapes, sfreq, channel names
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from strain.config import DREAMER_PROCESSED_DIR


def dreamer_processed_dir(base: Path | str | None = None) -> Path:
    return Path(base).resolve() if base is not None else DREAMER_PROCESSED_DIR


def dreamer_manifest_path(base: Path | None = None) -> Path:
    return dreamer_processed_dir(base) / "manifest.json"


def load_dreamer_manifest(base: Path | None = None) -> dict[str, Any]:
    mp = dreamer_manifest_path(base)
    if not mp.is_file():
        raise FileNotFoundError(
            f"No DREAMER manifest at {mp}. Run: python scripts/export_dreamer_epochs.py --mat ... "
        )
    return json.loads(mp.read_text())


def open_dreamer_X_memmap(base: Path | None = None, mode: str = "r") -> np.ndarray:
    """Open ``X.npy`` as read-only memory map (default ``mode='r'``)."""
    meta = load_dreamer_manifest(base)
    d = dreamer_processed_dir(base)
    path = d / meta["x_file"]
    if mode == "r":
        return np.load(str(path), mmap_mode="r")
    return np.lib.format.open_memmap(
        str(path),
        mode=mode,
        dtype=np.dtype(meta["dtype"]),
        shape=tuple(meta["x_shape"]),
    )


def load_dreamer_epoch_row(
    index: int,
    base: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one epoch tensor ``(14, T)`` and sidecar labels for indexing."""
    d = dreamer_processed_dir(base)
    meta = load_dreamer_manifest(base)
    X = open_dreamer_X_memmap(base, mode="r")
    i = index % X.shape[0]
    x = np.asarray(X[i], dtype=np.float32)
    side = {
        "subject_id": int(np.load(d / "subject_id.npy", mmap_mode="r")[i]),
        "trial_id": int(np.load(d / "trial_id.npy", mmap_mode="r")[i]),
        "start_sample": int(np.load(d / "start_sample.npy", mmap_mode="r")[i]),
        "valence": float(np.load(d / "valence.npy", mmap_mode="r")[i]),
        "arousal": float(np.load(d / "arousal.npy", mmap_mode="r")[i]),
        "dominance": float(np.load(d / "dominance.npy", mmap_mode="r")[i]),
        "sfreq": meta["sfreq"],
        "ch_names": meta["ch_names"],
    }
    return x, side
