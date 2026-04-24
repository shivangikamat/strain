"""Data Curator agent — loads tabular emotions dataset."""

from __future__ import annotations

from typing import Any

from emotiscan.io.emotions_csv import load_emotions_csv


class DataCurator:
    def load(
        self,
        csv_path: str | None = None,
        row_index: int = 0,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        ds = load_emotions_csv(csv_path, max_rows=max_rows)
        n = ds.X.shape[0]
        idx = row_index % n
        return {
            "row_index": idx,
            "feature_vector": ds.X[idx].tolist(),
            "label_true": str(ds.y[idx]),
            "feature_names": ds.feature_names,
            "metadata": ds.metadata,
            "path": str(ds.path),
        }
