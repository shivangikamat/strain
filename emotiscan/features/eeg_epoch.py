"""
Spectral features from a short EEG epoch ``(n_channels, n_times)`` — for DREAMER-style tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import signal


def _welch_psd_multichannel(
    eeg: np.ndarray,
    sfreq: float,
    *,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("eeg must be 2D (n_ch, n_times).")
    nper = nperseg or min(x.shape[1], 256)
    freqs, psd = signal.welch(x, fs=sfreq, axis=-1, nperseg=nper, noverlap=nper // 2)
    return freqs, psd


def _flat_per_channel_band_means(
    freqs: np.ndarray,
    psd: np.ndarray,
    channel_names: Sequence[str],
) -> dict[str, float]:
    """Keys like ``beta_AF3`` for Brain3D / per-electrode viz (same bands as global means)."""
    out: dict[str, float] = {}
    bands = (
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 31.0),
        ("gamma", 31.0, 45.0),
    )
    for ci, name in enumerate(channel_names):
        for bname, f0, f1 in bands:
            m = (freqs >= f0) & (freqs < f1)
            if not np.any(m):
                out[f"{bname}_{name}"] = 0.0
            else:
                out[f"{bname}_{name}"] = float(np.mean(psd[ci, m]))
    return out


def band_powers_welch(
    eeg: np.ndarray,
    sfreq: float,
    *,
    nperseg: int | None = None,
    channel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Welch PSD per channel; global band means average across channels.

    When ``channel_names`` is set and its length matches ``n_channels``, ``band_mean_power``
    also contains per-electrode keys (``theta_AF3``, ``beta_AF3``, …) for 3D scalp maps.
    """
    x = np.asarray(eeg, dtype=np.float64)
    freqs, psd = _welch_psd_multichannel(eeg, sfreq, nperseg=nperseg)

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
    bmp: dict[str, float] = {"theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma}
    if channel_names is not None:
        chs = list(channel_names)
        if len(chs) == x.shape[0]:
            bmp.update(_flat_per_channel_band_means(freqs, psd, chs))

    return {
        "differential_entropy": {
            "theta": 0.5 * math.log2(theta + eps),
            "alpha": 0.5 * math.log2(alpha + eps),
            "beta": 0.5 * math.log2(beta + eps),
            "gamma": 0.5 * math.log2(gamma + eps),
        },
        "band_mean_power": bmp,
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
    channel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Plan-shaped ``extract_features`` output for a raw epoch tensor."""
    bands = band_powers_welch(eeg, sfreq, channel_names=channel_names)
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
