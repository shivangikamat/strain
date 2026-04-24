"""Fixed pipeline: DataCurator → AnalysisEngine → Explainer."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from emotiscan.agents.analysis_engine import AnalysisEngine
from emotiscan.agents.data_curator import DataCurator
from emotiscan.agents.explainer import Explainer


def _row_index_from_query(query: str, modulo: int) -> int:
    m = re.search(r"row\s*[=:]?\s*(\d+)", query, re.I)
    if m:
        return int(m.group(1))
    h = int(hashlib.sha256(query.encode()).hexdigest(), 16)
    return h % max(modulo, 1)


class Orchestrator:
    def __init__(self) -> None:
        self.curator = DataCurator()
        self.engine = AnalysisEngine()
        self.explainer = Explainer()

    def run(
        self,
        query: str,
        *,
        csv_path: str | None = None,
    ) -> dict[str, Any]:
        from emotiscan.io.emotions_csv import load_emotions_csv

        ds = load_emotions_csv(csv_path, max_rows=None)
        idx = _row_index_from_query(query, ds.X.shape[0])
        loaded = self.curator.load(csv_path=csv_path, row_index=idx)
        analysis = self.engine.analyze_row(
            loaded["feature_vector"],
            loaded["feature_names"],
            with_explanation=True,
        )
        summary = self.explainer.summarize(analysis)
        return {
            "query": query,
            "row_index": loaded["row_index"],
            "ground_truth_label": loaded["label_true"],
            "analysis": analysis,
            "summary": summary,
        }
