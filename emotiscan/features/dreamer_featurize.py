"""Fixed-size feature vectors from DREAMER windows for sklearn / Ridge VAD."""

from __future__ import annotations

import numpy as np

from emotiscan.features.eeg_epoch import band_powers_welch


def featurize_dreamer_epoch(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Map ``(14, n_times)`` → 1D vector: band means, ratios, per-channel variance (14).
    """
    x = np.asarray(eeg, dtype=np.float64)
    b = band_powers_welch(x, sfreq)
    bm = b["band_mean_power"]
    vec = np.array(
        [
            bm["theta"],
            bm["alpha"],
            bm["beta"],
            bm["gamma"],
            b["spectral_ratios"]["theta_alpha"],
            b["spectral_ratios"]["beta_alpha"],
        ],
        dtype=np.float64,
    )
    ch_var = np.var(x, axis=1, dtype=np.float64)
    return np.concatenate([vec, ch_var], axis=0)


FEATURE_NAMES: tuple[str, ...] = (
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "theta_alpha_ratio",
    "beta_alpha_ratio",
    *[f"ch_var_{i}" for i in range(14)],
)
