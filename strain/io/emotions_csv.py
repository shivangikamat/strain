"""Load Kaggle EEG brainwave emotions CSV (tabular features + label)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EmotionsDataset:
    """In-memory view of emotions.csv."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    label_names: list[str]
    path: Path
    metadata: dict[str, Any]


def load_emotions_csv(
    path: Path | str | None = None,
    *,
    max_rows: int | None = None,
) -> EmotionsDataset:
    """
    Load ``data/emotions.csv`` (or ``path``): 2548 numeric feature columns + ``label``.

    The first header cell may be ``'# mean_0_a'``; it is normalized to ``mean_0_a``.
    """
    csv_path = Path(path) if path is not None else None
    if csv_path is None:
        from strain.config import EMOTIONS_CSV

        csv_path = EMOTIONS_CSV

    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Emotions CSV not found: {csv_path}. "
            "Set STRAIN_EMOTIONS_CSV or add data/emotions.csv."
        )

    df = pd.read_csv(csv_path, nrows=max_rows)
    df.columns = [c.lstrip("#").strip() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in emotions CSV.")

    y_raw = df["label"].astype(str).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float64)

    # Drop non-numeric columns if any
    non_numeric = df.drop(columns=["label"]).select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        X = df.drop(columns=["label", *non_numeric]).to_numpy(dtype=np.float64)
        feature_names = [c for c in df.columns if c != "label" and c not in non_numeric]
    else:
        feature_names = [c for c in df.columns if c != "label"]

    labels_sorted = sorted({str(x) for x in y_raw})
    metadata = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_distribution": {
            str(k): int(v) for k, v in pd.Series(y_raw).value_counts().items()
        },
        "note": (
            "Tabular engineered features (means, FFT bins, etc.), not raw EEG voltage. "
            "No subject ID in file — use stratified CV only; external validity is limited."
        ),
    }

    return EmotionsDataset(
        X=X,
        y=y_raw,
        feature_names=feature_names,
        label_names=labels_sorted,
        path=csv_path,
        metadata=metadata,
    )


def load_dataset(
    dataset: str = "eeg_brainwave",
    csv_path: Path | str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """
    Plan-aligned ``load_dataset`` for the Kaggle emotions export.

    ``dataset`` must be ``eeg_brainwave`` (emotions.csv). Returns JSON-serializable dict.
    """
    if dataset not in ("eeg_brainwave", "emotions_csv"):
        raise ValueError(f"Unsupported dataset: {dataset}. Use eeg_brainwave.")

    ds = load_emotions_csv(csv_path, max_rows=max_rows)
    return {
        "dataset": dataset,
        "path": str(ds.path),
        "n_samples": ds.metadata["n_samples"],
        "n_features": ds.metadata["n_features"],
        "labels": ds.label_names,
        "label_distribution": ds.metadata["label_distribution"],
        "feature_names_sample": ds.feature_names[:20],
        "metadata": ds.metadata,
    }
