"""
Find ideal DREAMER epoch indices for the 3 demo patient profiles.

Run after exporting DREAMER epochs and training the VAD model:
  python scripts/pick_demo_epochs.py

Prints recommended epoch indices. Copy them into DEMO_PATIENTS in api/main.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from strain.data.dreamer_epochs import (
    dreamer_processed_dir,
    load_dreamer_manifest,
    open_dreamer_X_memmap,
)
from strain.features.dreamer_featurize import featurize_dreamer_epoch
from strain.models.dreamer_vad import load_dreamer_vad_bundle

SAMPLE_EVERY = 5  # check every Nth epoch for speed


def _score_profile(v: float, a: float, d: float, profile: str) -> float:
    """Higher = better match for the profile."""
    if profile == "stressed":
        # High arousal, mid-low valence
        return a - abs(v - 2.8)
    if profile == "calm":
        # High valence, low arousal
        return v - a
    if profile == "executive":
        # High dominance, high arousal
        return d + a * 0.5
    return 0.0


def main() -> None:
    base = dreamer_processed_dir(None)
    meta = load_dreamer_manifest(base)
    sfreq = float(meta["sfreq"])
    X = open_dreamer_X_memmap(base, mode="r")
    n = int(X.shape[0])

    try:
        bundle = load_dreamer_vad_bundle()
    except FileNotFoundError:
        print("ERROR: VAD model not found. Run: python scripts/train_dreamer_vad.py")
        sys.exit(1)

    pipeline = bundle["pipeline"]
    v_all = np.load(str(base / "valence.npy"), mmap_mode="r")
    a_all = np.load(str(base / "arousal.npy"), mmap_mode="r")
    d_all = np.load(str(base / "dominance.npy"), mmap_mode="r")

    profiles = {"stressed": (-1, -999.0), "calm": (-1, -999.0), "executive": (-1, -999.0)}

    indices = range(0, n, SAMPLE_EVERY)
    print(f"Scanning {len(list(indices))} epochs (every {SAMPLE_EVERY} of {n})...")

    for i in indices:
        eeg = np.asarray(X[i], dtype=np.float64)
        feats = featurize_dreamer_epoch(eeg, sfreq).reshape(1, -1)
        pred = pipeline.predict(feats)[0]
        pv, pa, pd = float(pred[0]), float(pred[1]), float(pred[2])

        for name in profiles:
            score = _score_profile(pv, pa, pd, name)
            if score > profiles[name][1]:
                profiles[name] = (i, score)

    print("\n=== Recommended epoch indices ===")
    print(f"Alex Chen    (stressed):  epoch {profiles['stressed'][0]}")
    print(f"  true VAD: V={v_all[profiles['stressed'][0]]:.2f} "
          f"A={a_all[profiles['stressed'][0]]:.2f} "
          f"D={d_all[profiles['stressed'][0]]:.2f}")
    print()
    print(f"Maria Santos (calm):      epoch {profiles['calm'][0]}")
    print(f"  true VAD: V={v_all[profiles['calm'][0]]:.2f} "
          f"A={a_all[profiles['calm'][0]]:.2f} "
          f"D={d_all[profiles['calm'][0]]:.2f}")
    print()
    print(f"James O'Brien (executive): epoch {profiles['executive'][0]}")
    print(f"  true VAD: V={v_all[profiles['executive'][0]]:.2f} "
          f"A={a_all[profiles['executive'][0]]:.2f} "
          f"D={d_all[profiles['executive'][0]]:.2f}")
    print()
    print("Copy these indices into DEMO_PATIENTS in api/main.py")


if __name__ == "__main__":
    main()
