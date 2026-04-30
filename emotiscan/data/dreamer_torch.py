"""
PyTorch ``Dataset`` over exported DREAMER memmaps.

Requires ``torch`` (project dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from emotiscan.data.dreamer_epochs import (
    dreamer_processed_dir,
    load_dreamer_manifest,
    open_dreamer_X_memmap,
)


class DreamerEpochDataset(Dataset):
    """
    Yields ``eeg`` ``float32`` tensor ``(14, n_times)``, ``vad`` tensor ``(3,)`` (valence, arousal, dominance),
    plus ``subject_id``, ``trial_id``, ``epoch_id``.

    Pass ``indices`` (1D int64) to restrict to a subject-holdout split.
    """

    def __init__(
        self,
        *,
        processed_dir: Path | str | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self._base = dreamer_processed_dir(
            Path(processed_dir).resolve() if processed_dir is not None else None
        )
        self.meta = load_dreamer_manifest(self._base)
        self.X = open_dreamer_X_memmap(self._base, mode="r")
        n = self.X.shape[0]
        self._idx = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(n, dtype=np.int64)
        )
        d = self._base
        self._v = np.load(str(d / "valence.npy"), mmap_mode="r")
        self._a = np.load(str(d / "arousal.npy"), mmap_mode="r")
        self._d = np.load(str(d / "dominance.npy"), mmap_mode="r")
        self._s = np.load(str(d / "subject_id.npy"), mmap_mode="r")
        self._t = np.load(str(d / "trial_id.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def __getitem__(self, i: int) -> dict[str, Any]:
        j = int(self._idx[i])
        x = np.asarray(self.X[j], dtype=np.float32)
        eeg = torch.from_numpy(x.copy())
        vad = torch.tensor(
            [float(self._v[j]), float(self._a[j]), float(self._d[j])],
            dtype=torch.float32,
        )
        return {
            "eeg": eeg,
            "vad": vad,
            "subject_id": int(self._s[j]),
            "trial_id": int(self._t[j]),
            "epoch_id": j,
        }
