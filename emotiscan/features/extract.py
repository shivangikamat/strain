"""
Feature extraction for emotions.csv rows.

True hemispheric asymmetry is not available (single derived feature set).
We aggregate FFT columns into four proxy bands by bin index (low → high frequency).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _fft_column_indices(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith("fft_")]


def fft_band_proxy_features(
    feature_vector: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Split FFT bins into four equal-count bands → proxy theta/alpha/beta/gamma energy (mean abs value).
    """
    idx = _fft_column_indices(feature_names)
    if not idx:
        return {
            "theta_proxy": 0.0,
            "alpha_proxy": 0.0,
            "beta_proxy": 0.0,
            "gamma_proxy": 0.0,
        }

    values = np.abs(np.asarray(feature_vector, dtype=np.float64)[idx])
    n = len(values)
    q = n // 4
    bands = [
        values[:q] if q else values,
        values[q : 2 * q] if q else values,
        values[2 * q : 3 * q] if q else values,
        values[3 * q :] if q else values,
    ]
    names = ("theta_proxy", "alpha_proxy", "beta_proxy", "gamma_proxy")
    return {name: float(np.mean(part)) if len(part) else 0.0 for name, part in zip(names, bands)}


def differential_entropy_proxy(band_values: dict[str, float]) -> dict[str, float]:
    """Map band energy to a DE-like score: 0.5 * log2(1 + energy)."""
    return {k: 0.5 * math.log2(1.0 + max(v, 0.0)) for k, v in band_values.items()}


def extract_features(
    feature_vector: np.ndarray,
    feature_names: list[str],
    *,
    include_full_vector_stats: bool = True,
) -> dict[str, Any]:
    """
    Plan-aligned ``extract_features`` for tabular Kaggle rows.

    Returns band proxies, DE proxies, simple temporal stats on the vector (not time series).
    """
    x = np.asarray(feature_vector, dtype=np.float64).ravel()
    bands = fft_band_proxy_features(x, feature_names)
    de = differential_entropy_proxy(bands)

    theta = bands["theta_proxy"]
    alpha = max(bands["alpha_proxy"], 1e-9)
    beta = bands["beta_proxy"]

    out: dict[str, Any] = {
        "differential_entropy": de,
        "band_energy_proxy": bands,
        "spectral_ratios": {
            "theta_alpha": float(theta / alpha),
            "beta_alpha": float(beta / alpha),
        },
        "hemispheric_features": {
            "note": "Not available for this dataset (no left/right channels in CSV).",
            "asymmetry_index": None,
            "frontal_alpha_asymmetry": None,
        },
    }

    if include_full_vector_stats:
        out["vector_stats"] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return out
