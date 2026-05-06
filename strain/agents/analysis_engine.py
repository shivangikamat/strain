"""Analysis Engine — features, classification, screening."""

from __future__ import annotations

from typing import Any

import numpy as np

from strain.features.extract import extract_features
from strain.models.classifier import classify_emotion, explain_decision, load_classifier_pipeline
from strain.screening.mental_health import screen_mental_health


class AnalysisEngine:
    def __init__(self, bundle: dict | None = None):
        self._bundle = bundle

    def analyze_row(
        self,
        feature_vector: list[float],
        feature_names: list[str],
        *,
        with_explanation: bool = True,
    ) -> dict[str, Any]:
        x = np.asarray(feature_vector, dtype=np.float64)
        feats = extract_features(x, feature_names)
        bundle = self._bundle or load_classifier_pipeline()
        cls = classify_emotion(x, feature_names=feature_names, bundle=bundle)
        screen = screen_mental_health(cls, feats)
        out: dict[str, Any] = {
            "features": feats,
            "classification": cls,
            "screening": screen,
        }
        if with_explanation:
            out["explanation"] = explain_decision(x, feature_names, cls, bundle=bundle)
        return out
