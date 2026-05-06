"""Subject-wise train/test splits for DREAMER epoch tensors (avoid leakage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from strain.data.dreamer_epochs import dreamer_processed_dir, load_dreamer_manifest


def load_subject_ids_memmap(base: Path | None = None) -> np.ndarray:
    d = dreamer_processed_dir(base)
    return np.load(str(d / "subject_id.npy"), mmap_mode="r")


def train_test_mask_by_subject(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    base: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Return boolean masks ``train_m``, ``test_m`` over epoch axis, grouped by ``subject_id``.

    Same subject never appears in both train and test.
    """
    meta = load_dreamer_manifest(base)
    n = int(meta["n_epochs"])
    groups = np.asarray(load_subject_ids_memmap(base)[:n], dtype=np.int32)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.zeros((n, 1)), groups=groups))
    train_m = np.zeros(n, dtype=bool)
    test_m = np.zeros(n, dtype=bool)
    train_m[train_idx] = True
    test_m[test_idx] = True
    info = {
        "n_epochs": n,
        "n_train": int(train_m.sum()),
        "n_test": int(test_m.sum()),
        "n_subjects": int(np.unique(groups).size),
        "test_subjects": sorted(int(x) for x in np.unique(groups[test_m]).tolist()),
        "train_subjects": sorted(int(x) for x in np.unique(groups[train_m]).tolist()),
    }
    return train_m, test_m, info
