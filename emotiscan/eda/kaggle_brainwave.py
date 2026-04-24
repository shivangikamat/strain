"""
EDA for Kaggle emotions.csv (tabular EEG-derived features).

Run: python -m emotiscan.eda.kaggle_brainwave
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotiscan.config import EMOTIONS_CSV  # noqa: E402
from emotiscan.features.extract import extract_features  # noqa: E402
from emotiscan.io.emotions_csv import load_emotions_csv  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EMOTIONS_CSV
    print(f"Loading {path} ...")
    ds = load_emotions_csv(path)

    print("\n=== Summary ===")
    print(json.dumps(ds.metadata, indent=2))

    # Per-label vector mean (first 5 features only for display)
    df = pd.DataFrame(ds.X[:, :5], columns=ds.feature_names[:5])
    df["label"] = ds.y
    print("\n=== Mean of first 5 features by label ===")
    print(df.groupby("label").mean())

    # Example proxy features for one row per class
    print("\n=== Example extract_features (one row per class) ===")
    for lab in sorted(set(ds.y)):
        idx = int(np.where(ds.y == lab)[0][0])
        feats = extract_features(ds.X[idx], ds.feature_names)
        print(lab, json.dumps({k: feats[k] for k in ("spectral_ratios", "band_energy_proxy")}, indent=2))

    # Stratified CV baseline (no subject grouping — not in CSV)
    le = LabelEncoder()
    y = le.fit_transform(ds.y)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, ds.X, y, cv=cv, scoring="accuracy")
    print("\n=== Stratified 5-fold accuracy (no subject ID — limited external validity) ===")
    print("scores:", scores)
    print("mean:", float(np.mean(scores)), "std:", float(np.std(scores)))

    y_pred = np.zeros_like(y)
    for tr, te in cv.split(ds.X, y):
        clf.fit(ds.X[tr], y[tr])
        y_pred[te] = clf.predict(ds.X[te])
    print("\n=== Classification report (CV predictions) ===")
    print(classification_report(y, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
