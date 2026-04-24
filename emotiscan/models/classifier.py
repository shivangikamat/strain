"""Baseline sklearn pipeline: StandardScaler + LogisticRegression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from emotiscan.config import BASELINE_PIPELINE_PATH, DEFAULT_RANDOM_STATE, MODEL_DIR
from emotiscan.io.emotions_csv import load_emotions_csv


def train_and_save_baseline(
    csv_path: Path | str | None = None,
    *,
    out_path: Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train pipeline on full CSV, 5-fold stratified CV metrics, save joblib."""
    ds = load_emotions_csv(csv_path)
    le = LabelEncoder()
    y = le.fit_transform(ds.y)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=4000,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, ds.X, y, cv=cv)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)

    clf.fit(ds.X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path or BASELINE_PIPELINE_PATH
    joblib.dump({"pipeline": clf, "label_encoder": le, "feature_names": ds.feature_names}, path)

    return {
        "model_path": str(path),
        "cv_folds": 5,
        "classification_report": report,
        "classes": le.classes_.tolist(),
    }


def load_classifier_pipeline(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else BASELINE_PIPELINE_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing trained model at {p}. Run: python -m scripts.train_baseline"
        )
    bundle = joblib.load(p)
    return bundle


def predict_row(
    feature_vector: np.ndarray,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return class, probabilities, confidence."""
    b = bundle or load_classifier_pipeline()
    clf: Pipeline = b["pipeline"]
    le: LabelEncoder = b["label_encoder"]
    x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return {
        "discrete_emotion": str(le.classes_[idx]),
        "probabilities": {str(c): float(p) for c, p in zip(le.classes_, proba)},
        "confidence": float(proba[idx]),
    }


def classify_emotion(
    feature_vector: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan-aligned wrapper (tabular features). valence/arousal are heuristic from probs."""
    x = np.asarray(feature_vector, dtype=np.float64).ravel()
    pred = predict_row(x, bundle=bundle)
    probs = pred["probabilities"]
    # Heuristic mapping for demo UI only
    v_neg = probs.get("NEGATIVE", 0.0)
    v_pos = probs.get("POSITIVE", 0.0)
    valence = float(v_pos - v_neg)
    arousal = float(1.0 - probs.get("NEUTRAL", 0.0))
    return {
        **pred,
        "valence": max(-1.0, min(1.0, valence)),
        "arousal": max(0.0, min(1.0, arousal)),
    }


def explain_decision(
    feature_vector: np.ndarray,
    feature_names: list[str],
    prediction: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    """
    Attribution via logistic regression coefficients (global linear model), not LIME.
    """
    b = bundle or load_classifier_pipeline()
    clf: Pipeline = b["pipeline"]
    le: LabelEncoder = b["label_encoder"]
    lr: LogisticRegression = clf.named_steps["lr"]
    scaler: StandardScaler = clf.named_steps["scaler"]

    x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    xs = scaler.transform(x)[0]

    target = prediction.get("discrete_emotion")
    probs = prediction.get("probabilities") or {}
    if target not in le.classes_:
        if probs:
            target = max(probs, key=probs.get)
        else:
            target = str(le.classes_[0])

    class_idx = int(np.where(le.classes_ == target)[0][0])
    coef = lr.coef_[class_idx]
    contrib = coef * xs
    order = np.argsort(np.abs(contrib))[::-1][:top_k]

    top_features = [
        {
            "name": feature_names[i],
            "contribution": float(contrib[i]),
            "scaled_input": float(xs[i]),
        }
        for i in order
    ]

    nl = (
        f"Top drivers for '{target}' (linear model): "
        + ", ".join(f"{t['name']} ({t['contribution']:+.3f})" for t in top_features[:5])
        + ". Prototype explanation only."
    )

    return {
        "predicted_class": target,
        "channel_importance": {"note": "N/A — tabular features, not per-electrode."},
        "frequency_importance": {"note": "See top feature names (many are fft_* bins)."},
        "top_features": top_features,
        "natural_language_explanation": nl,
    }


