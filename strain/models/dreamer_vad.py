"""Multi-output regression: band features → valence / arousal / dominance (DREAMER)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strain.config import DEFAULT_RANDOM_STATE, MODEL_DIR
from strain.data.dreamer_epochs import dreamer_processed_dir, open_dreamer_X_memmap
from strain.data.dreamer_splits import train_test_mask_by_subject
from strain.features.dreamer_featurize import FEATURE_NAMES, featurize_dreamer_epoch

DREAMER_VAD_PIPELINE_PATH = MODEL_DIR / "dreamer_vad_multiridge.joblib"


def _stack_features(
    base: Path,
    X_mm: np.ndarray,
    sfreq: float,
    row_indices: np.ndarray,
    *,
    max_samples: int | None = None,
    log_every: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build design matrix and VAD targets for given epoch indices."""
    ddir = dreamer_processed_dir(base)
    v_all = np.load(str(ddir / "valence.npy"), mmap_mode="r")
    a_all = np.load(str(ddir / "arousal.npy"), mmap_mode="r")
    d_all = np.load(str(ddir / "dominance.npy"), mmap_mode="r")

    idx = np.asarray(row_indices, dtype=np.int64)
    if max_samples is not None and idx.size > max_samples:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
        idx = rng.choice(idx, size=max_samples, replace=False)

    feats: list[np.ndarray] = []
    yv: list[float] = []
    ya: list[float] = []
    yd: list[float] = []
    for k, j in enumerate(idx):
        j = int(j)
        if log_every and k > 0 and k % log_every == 0:
            print(f"  featurize {k}/{idx.size}")
        eeg = np.asarray(X_mm[j], dtype=np.float64)
        feats.append(featurize_dreamer_epoch(eeg, sfreq))
        yv.append(float(v_all[j]))
        ya.append(float(a_all[j]))
        yd.append(float(d_all[j]))

    Xf = np.stack(feats, axis=0)
    y = np.column_stack([yv, ya, yd]).astype(np.float64)
    return Xf, y, idx


def train_and_save_dreamer_vad(
    *,
    processed_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    out_path: Path | None = None,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    """
    Train ``StandardScaler`` + ``MultiOutputRegressor(Ridge)`` with **subject-grouped** holdout.
    """
    from strain.data.dreamer_epochs import load_dreamer_manifest

    base = (
        Path(processed_dir).resolve()
        if processed_dir is not None
        else dreamer_processed_dir(None)
    )
    meta = load_dreamer_manifest(base)
    sfreq = float(meta["sfreq"])
    X_mm = open_dreamer_X_memmap(base, mode="r")
    n = X_mm.shape[0]
    train_m, test_m, split_info = train_test_mask_by_subject(
        test_size=test_size, random_state=random_state, base=base
    )
    train_idx = np.where(train_m)[0]
    test_idx = np.where(test_m)[0]

    print(f"Building train features ({train_idx.size} epochs)…")
    X_tr, y_tr, _ = _stack_features(
        base, X_mm, sfreq, train_idx, max_samples=max_train_samples
    )
    print(f"Building test features ({test_idx.size} epochs)…")
    X_te, y_te, _ = _stack_features(
        base, X_mm, sfreq, test_idx, max_samples=max_train_samples
    )

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ridge",
                MultiOutputRegressor(Ridge(alpha=2.0, random_state=random_state)),
            ),
        ]
    )
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    mae = mean_absolute_error(y_te, pred, multioutput="raw_values")
    mae_dict = {
        "valence": float(mae[0]),
        "arousal": float(mae[1]),
        "dominance": float(mae[2]),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path or DREAMER_VAD_PIPELINE_PATH
    bundle = {
        "pipeline": reg,
        "feature_names": list(FEATURE_NAMES),
        "sfreq": sfreq,
        "split": split_info,
        "test_mae": mae_dict,
        "processed_dir": str(base.resolve()),
    }
    joblib.dump(bundle, path)

    return {
        "model_path": str(path),
        "test_mae": mae_dict,
        "split": split_info,
        "n_train_epochs_used": int(X_tr.shape[0]),
        "n_test_epochs_used": int(X_te.shape[0]),
    }


def load_dreamer_vad_bundle(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DREAMER_VAD_PIPELINE_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"No DREAMER VAD model at {p}. Run: python scripts/train_dreamer_vad.py"
        )
    return joblib.load(p)


def predict_vad(eeg: np.ndarray, sfreq: float, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    b = bundle or load_dreamer_vad_bundle()
    x = featurize_dreamer_epoch(eeg, sfreq).reshape(1, -1)
    y = b["pipeline"].predict(x)[0]
    return {
        "valence": float(y[0]),
        "arousal": float(y[1]),
        "dominance": float(y[2]),
    }


def explain_vad_ridge(
    eeg: np.ndarray,
    sfreq: float,
    true_vad: dict[str, float],
    pred_vad: dict[str, float],
    bundle: dict[str, Any] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Per-target linear attribution from scaled Ridge coefficients."""
    b = bundle or load_dreamer_vad_bundle()
    pipe: Pipeline = b["pipeline"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    mor: MultiOutputRegressor = pipe.named_steps["ridge"]
    names = b["feature_names"]
    x = featurize_dreamer_epoch(eeg, sfreq).reshape(1, -1)
    xs = scaler.transform(x)[0]

    targets = ("valence", "arousal", "dominance")
    per_target: dict[str, Any] = {}
    for ti, tname in enumerate(targets):
        est = mor.estimators_[ti]
        coef = est.coef_
        contrib = coef * xs
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        per_target[tname] = [
            {"name": names[i], "contribution": float(contrib[i])} for i in order
        ]

    nl = (
        f"Demo VAD: true V={true_vad['valence']:.2f} A={true_vad['arousal']:.2f} D={true_vad['dominance']:.2f}; "
        f"pred V={pred_vad['valence']:.2f} A={pred_vad['arousal']:.2f} D={pred_vad['dominance']:.2f}. "
        "Ridge on band + channel-variance features (prototype)."
    )
    return {
        "per_target_top_features": per_target,
        "natural_language_explanation": nl,
    }


def dreamer_vad_screening(pred: dict[str, float], true_v: dict[str, float] | None = None) -> dict[str, Any]:
    """Map continuous VAD to demo risk copy (non-clinical)."""
    v = pred["valence"]
    a = pred["arousal"]
    dep = max(0.0, min(100.0, (3.0 - v) * 25.0 + max(0.0, a - 3.0) * 10.0))
    anx = max(0.0, min(100.0, (a - 2.5) * 22.0 + abs(v - 3.0) * 5.0))
    rec = "no_concern"
    if dep > 65 or anx > 65:
        rec = "monitor"
    if dep > 80 or anx > 80:
        rec = "consult_pcp"
    return {
        "disclaimer": "Demonstration mapping from model VAD only — not clinical.",
        "depression_risk": {"score": dep, "note": "Higher when predicted valence is low."},
        "anxiety_risk": {"score": anx, "note": "Higher when predicted arousal is high."},
        "recommendation": rec,
        "key_findings": [
            f"Predicted valence/arousal/dominance: {v:.2f} / {a:.2f} / {pred['dominance']:.2f}",
        ],
    }
