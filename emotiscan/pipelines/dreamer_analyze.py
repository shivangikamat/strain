"""End-to-end analysis for one DREAMER epoch (features → VAD → explanation → demo screening)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emotiscan.data.dreamer_epochs import load_dreamer_epoch_row, open_dreamer_X_memmap
from emotiscan.features.eeg_epoch import extract_features_from_epoch
from emotiscan.models.dreamer_vad import (
    dreamer_vad_screening,
    explain_vad_ridge,
    load_dreamer_vad_bundle,
    predict_vad,
)


def dreamer_epoch_count(processed_dir: Path | str | None = None) -> int:
    X = open_dreamer_X_memmap(processed_dir, mode="r")
    return int(X.shape[0])


def analyze_dreamer_epoch(
    epoch_index: int,
    processed_dir: Path | str | None = None,
) -> dict[str, Any]:
    X = open_dreamer_X_memmap(processed_dir, mode="r")
    n = int(X.shape[0])
    idx = epoch_index % n
    eeg, side = load_dreamer_epoch_row(idx, processed_dir)
    sfreq = float(side["sfreq"])
    feats = extract_features_from_epoch(eeg, sfreq)
    true_vad = {
        "valence": float(side["valence"]),
        "arousal": float(side["arousal"]),
        "dominance": float(side["dominance"]),
    }

    pred_vad = None
    explanation: dict[str, Any] = {
        "natural_language_explanation": (
            "VAD regressor not trained. Run: python scripts/train_dreamer_vad.py "
            "(requires exported DREAMER epochs)."
        ),
        "per_target_top_features": {},
    }
    screening = None

    try:
        bundle = load_dreamer_vad_bundle()
        pred_vad = predict_vad(eeg, sfreq, bundle=bundle)
        explanation = explain_vad_ridge(
            eeg,
            sfreq,
            true_vad,
            pred_vad,
            bundle=bundle,
        )
        screening = dreamer_vad_screening(pred_vad)
    except FileNotFoundError:
        pass

    return {
        "epoch_index": idx,
        "subject_id": side["subject_id"],
        "trial_id": side["trial_id"],
        "true_vad": true_vad,
        "features": feats,
        "predicted_vad": pred_vad,
        "explanation": explanation,
        "screening": screening,
    }
