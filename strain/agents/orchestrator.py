"""Fixed pipeline: DataCurator → AnalysisEngine → Explainer (CSV) or DREAMER epoch pipeline."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from strain.agents.analysis_engine import AnalysisEngine
from strain.agents.data_curator import DataCurator
from strain.agents.explainer import Explainer


def _row_index_from_query(query: str, modulo: int) -> int:
    m = re.search(r"(?:row|epoch)\s*[=:]?\s*(\d+)", query, re.I)
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
        source: Literal["csv", "dreamer"] = "csv",
        csv_path: str | None = None,
        dreamer_processed_dir: str | None = None,
    ) -> dict[str, Any]:
        if source == "dreamer":
            from strain.pipelines.dreamer_analyze import (
                analyze_dreamer_epoch,
                dreamer_epoch_count,
            )

            n = dreamer_epoch_count(dreamer_processed_dir)
            idx = _row_index_from_query(query, n)
            analysis = analyze_dreamer_epoch(idx, processed_dir=dreamer_processed_dir)
            summary = analysis["explanation"].get(
                "natural_language_explanation",
                "DREAMER epoch analysis.",
            )
            return {
                "query": query,
                "source": "dreamer",
                "epoch_index": analysis["epoch_index"],
                "subject_id": analysis["subject_id"],
                "trial_id": analysis["trial_id"],
                "analysis": analysis,
                "summary": summary,
            }

        from strain.io.emotions_csv import load_emotions_csv

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
            "source": "csv",
            "row_index": loaded["row_index"],
            "ground_truth_label": loaded["label_true"],
            "analysis": analysis,
            "summary": summary,
        }
