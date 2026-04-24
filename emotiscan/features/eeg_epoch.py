"""
Spectral features from a short EEG epoch ``(n_channels, n_times)`` — for DREAMER-style tensors.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal


def band_powers_welch(
    eeg: np.ndarray,
    sfreq: float,
    *,
    nperseg: int | None = None,
) -> dict[str, Any]:
    """
    Average Welch PSD across channels, integrate standard bands (theta/alpha/beta/gamma).
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("eeg must be 2D (n_ch, n_times).")
    nper = nperseg or min(x.shape[1], 256)
    freqs, psd = signal.welch(x, fs=sfreq, axis=-1, nperseg=nper, noverlap=nper // 2)

    def band_mean(f0: float, f1: float) -> float:
        m = (freqs >= f0) & (freqs < f1)
        if not np.any(m):
            return 0.0
        return float(np.mean(psd[:, m]))

    theta = band_mean(4.0, 8.0)
    alpha = band_mean(8.0, 13.0)
    beta = band_mean(13.0, 31.0)
    gamma = band_mean(31.0, 45.0)

    eps = 1e-9
    return {
        "differential_entropy": {
            "theta": 0.5 * math.log2(theta + eps),
            "alpha": 0.5 * math.log2(alpha + eps),
            "beta": 0.5 * math.log2(beta + eps),
            "gamma": 0.5 * math.log2(gamma + eps),
        },
        "band_mean_power": {"theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma},
        "spectral_ratios": {
            "theta_alpha": float(theta / (alpha + eps)),
            "beta_alpha": float(beta / (alpha + eps)),
        },
        "hemispheric_features": {
            "note": "Use full 14-ch layout for asymmetry in a follow-up (Fp1/Fp2-style pairs).",
            "asymmetry_index": None,
            "frontal_alpha_asymmetry": None,
        },
    }


def extract_features_from_epoch(
    eeg: np.ndarray,
    sfreq: float = 128.0,
    *,
    include_vector_stats: bool = True,
) -> dict[str, Any]:
    """Plan-shaped ``extract_features`` output for a raw epoch tensor."""
    bands = band_powers_welch(eeg, sfreq)
    out: dict[str, Any] = {
        **bands,
        "band_energy_proxy": bands["band_mean_power"],
    }
    if include_vector_stats:
        xf = np.asarray(eeg, dtype=np.float64).ravel()
        out["vector_stats"] = {
            "mean": float(np.mean(xf)),
            "std": float(np.std(xf)),
            "min": float(np.min(xf)),
            "max": float(np.max(xf)),
        }
    return out
