"""
Load DREAMER.mat (Katsigiannis & Ramzan, 2017).

Open download: https://zenodo.org/records/546113 (place as ``data/raw/DREAMER.mat``).

Layout (``scipy.io.loadmat``, ``struct_as_record=False``):

- ``mat['DREAMER'][0, 0]['Data'][0, subject]`` — per-subject struct
- ``['EEG'][0,0]['stimuli'][0,0][trial, 0]`` — ``(n_samples, 14)`` float
- ``['ScoreValence'][0,0][trial, 0]`` (and Arousal, Dominance) — Likert 1–5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.io import loadmat


def _dreamer_field(obj: Any, key: str) -> Any:
    """
    Read a field from ``loadmat(..., struct_as_record=False)`` output.

    Top-level ``mat`` is a ``dict``; nested MATLAB structs are ``mat_struct`` and
    require attribute access — ``obj[key]`` raises TypeError on recent SciPy.
    """
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _unwrap_matlab_cell(cell: Any) -> Any:
    """
    MATLAB often stores one struct inside ``(1,1)`` or ``(1,)`` ``dtype=object`` ndarrays.
    Follow those wrappers until we hit a ``mat_struct`` or non-wrapper array.
    """
    cur: Any = cell
    while isinstance(cur, np.ndarray) and cur.dtype == object and cur.size == 1:
        cur = cur.flat[0]
    return cur


def _eeg_struct(sub: Any) -> Any:
    """Return the inner ``EEG`` ``mat_struct`` for one subject (unwraps ``(1,1)`` cells)."""
    eeg_cell = _dreamer_field(sub, "EEG")
    eeg = eeg_cell[0, 0] if isinstance(eeg_cell, np.ndarray) else eeg_cell
    return _unwrap_matlab_cell(eeg)


DREAMER_CHANNEL_NAMES: tuple[str, ...] = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)


@dataclass
class DreamerClip:
    """One window: shape ``(n_channels, n_times)``."""

    eeg: np.ndarray  # float32, (14, chunk_size)
    subject_id: int
    trial_id: int
    start_sample: int
    end_sample: int
    valence: float
    arousal: float
    dominance: float


def load_dreamer_mat(path: Path | str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"DREAMER.mat not found: {p}\n"
            "Download from https://zenodo.org/records/546113 (DREAMER.mat), "
            "save it (e.g. mkdir -p data/raw && mv DREAMER.mat data/raw/), "
            "or set EMOTISCAN_DREAMER_MAT to the full path and pass --mat \"$EMOTISCAN_DREAMER_MAT\"."
        )
    return loadmat(str(p), squeeze_me=False, struct_as_record=False)


def dreamer_root(mat: dict[str, Any]) -> Any:
    if "DREAMER" not in mat:
        raise KeyError("MAT file missing top-level 'DREAMER' struct.")
    return mat["DREAMER"][0, 0]


def dreamer_num_subjects(root: Any) -> int:
    data = _dreamer_field(root, "Data")
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return int(max(data.shape))
    return int(data.shape[0])


def _subject_struct(root: Any, subject_id: int) -> Any:
    data = _dreamer_field(root, "Data")
    if isinstance(data, np.ndarray) and data.ndim == 2:
        if data.shape[0] == 1:
            raw = data[0, subject_id]
        else:
            raw = data[subject_id, 0]
        return _unwrap_matlab_cell(raw)
    raise TypeError("Unexpected DREAMER.Data layout.")


def stimulus_array(root: Any, subject_id: int, trial_id: int) -> np.ndarray:
    """Return ``(n_samples, 14)`` float array for one movie trial."""
    sub = _subject_struct(root, subject_id)
    eeg = _eeg_struct(sub)
    stimuli = _dreamer_field(eeg, "stimuli")
    raw = stimuli[trial_id, 0]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 14:
        raise ValueError(f"Unexpected stimulus shape for s{subject_id} t{trial_id}: {arr.shape}")
    return arr[:, :14]


def scores_for_trial(root: Any, subject_id: int, trial_id: int) -> tuple[float, float, float]:
    sub = _subject_struct(root, subject_id)
    v = float(np.asarray(_dreamer_field(sub, "ScoreValence")[trial_id, 0]).squeeze())
    a = float(np.asarray(_dreamer_field(sub, "ScoreArousal")[trial_id, 0]).squeeze())
    d = float(np.asarray(_dreamer_field(sub, "ScoreDominance")[trial_id, 0]).squeeze())
    return v, a, d


def iter_dreamer_clips(
    mat: dict[str, Any],
    *,
    chunk_size: int = 256,
    overlap: int = 128,
    num_channel: int = 14,
    apply_filter: bool = True,
    sfreq: float = 128.0,
) -> Iterator[DreamerClip]:
    """
    Yield sliding-window clips (channels first) across all subjects and trials.

    Default: 256-sample windows (~2 s @ 128 Hz) with 50% overlap — strong default for
    fixed-shape CNN / Conformer inputs on DREAMER without padding variable-length trials.
    """
    import mne

    root = dreamer_root(mat)
    n_sub = dreamer_num_subjects(root)
    for subject_id in range(n_sub):
        sub = _subject_struct(root, subject_id)
        eeg = _eeg_struct(sub)
        stimuli = _dreamer_field(eeg, "stimuli")
        n_trials = int(stimuli.shape[0])
        for trial_id in range(n_trials):
            sig = stimulus_array(root, subject_id, trial_id)
            x = np.ascontiguousarray(sig[:, :num_channel].T, dtype=np.float64)
            if apply_filter:
                x = mne.filter.filter_data(x, sfreq, l_freq=1.0, h_freq=45.0, verbose=False)
            v, aro, dom = scores_for_trial(root, subject_id, trial_id)

            step = chunk_size - overlap
            if step <= 0:
                raise ValueError("chunk_size must be greater than overlap.")
            t = x.shape[1]
            start = 0
            while start + chunk_size <= t:
                end = start + chunk_size
                clip = x[:, start:end].astype(np.float32, copy=False)
                yield DreamerClip(
                    eeg=clip,
                    subject_id=subject_id,
                    trial_id=trial_id,
                    start_sample=start,
                    end_sample=end,
                    valence=v,
                    arousal=aro,
                    dominance=dom,
                )
                start += step


def count_dreamer_clips(mat: dict[str, Any], **kwargs: Any) -> int:
    return sum(1 for _ in iter_dreamer_clips(mat, **kwargs))
