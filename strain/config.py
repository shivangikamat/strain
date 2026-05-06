"""Paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

# Package lives at repo_root/strain/
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Primary Kaggle export: place at repo_root/data/emotions.csv (override with env).
EMOTIONS_CSV = Path(
    os.environ.get("STRAIN_EMOTIONS_CSV", _REPO_ROOT / "data" / "emotions.csv")
)

# DREAMER: raw MATLAB bundle (Zenodo / IEEE DataPort).
DREAMER_MAT = Path(os.environ.get("STRAIN_DREAMER_MAT", _REPO_ROOT / "data" / "raw" / "DREAMER.mat"))

# Preprocessed sliding-window tensors + labels (see scripts/export_dreamer_epochs.py).
DREAMER_PROCESSED_DIR = Path(
    os.environ.get("STRAIN_DREAMER_PROCESSED", _REPO_ROOT / "data" / "processed" / "dreamer")
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
BASELINE_PIPELINE_PATH = MODEL_DIR / "baseline_pipeline.joblib"

# Dataset has no explicit subject column; use stratified CV only.
DEFAULT_RANDOM_STATE = 42

LABEL_ORDER = ("NEGATIVE", "NEUTRAL", "POSITIVE")
