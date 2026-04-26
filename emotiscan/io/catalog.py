"""Unified dataset metadata (Kaggle CSV + DREAMER exports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dataset_meta(
    dataset: str,
    *,
    csv_path: str | Path | None = None,
    max_rows: int | None = None,
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    ``eeg_brainwave`` / ``emotions_csv`` → tabular Kaggle summary.
    ``dreamer`` → processed DREAMER ``manifest.json`` plus ``dataset`` key.
    """
    if dataset in ("eeg_brainwave", "emotions_csv"):
        from emotiscan.io.emotions_csv import load_dataset

        return load_dataset(dataset, csv_path=csv_path, max_rows=max_rows)

    if dataset == "dreamer":
        from emotiscan.data.dreamer_epochs import load_dreamer_manifest

        base = Path(processed_dir).resolve() if processed_dir is not None else None
        m = load_dreamer_manifest(base)
        out = dict(m)
        out["dataset"] = "dreamer"
        return out

    raise ValueError(f"Unknown dataset: {dataset}. Use eeg_brainwave or dreamer.")
