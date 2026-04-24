#!/usr/bin/env python3
"""
Export DREAMER.mat → memmapped epoch tensors for PyTorch / MNE workflows.

Strategy (recommended for this project):
  - **256-sample windows @ 128 Hz** (~2 s) with **50% overlap** (128 samples).
  - **Per-clip VAD labels** copied from the parent trial (standard for chunk-level training).
  - **1–45 Hz bandpass** via MNE (matches downstream spectral biomarker story).

Outputs (under ``data/processed/dreamer/`` by default):
  ``X.npy`` memmap + parallel ``*.npy`` index arrays + ``manifest.json``.

Usage:
  python scripts/export_dreamer_epochs.py --mat data/raw/DREAMER.mat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotiscan.config import DREAMER_PROCESSED_DIR  # noqa: E402
from emotiscan.io.dreamer_mat import (  # noqa: E402
    DREAMER_CHANNEL_NAMES,
    count_dreamer_clips,
    iter_dreamer_clips,
    load_dreamer_mat,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DREAMER.mat to memmapped epoch tensors.")
    ap.add_argument("--mat", type=Path, default=None, help="Path to DREAMER.mat")
    ap.add_argument("--out", type=Path, default=DREAMER_PROCESSED_DIR, help="Output directory")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--no-filter", action="store_true", help="Skip 1–45 Hz bandpass")
    ap.add_argument("--sfreq", type=float, default=128.0)
    args = ap.parse_args()

    mat_path = args.mat
    if mat_path is None:
        from emotiscan.config import DREAMER_MAT

        mat_path = DREAMER_MAT

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {mat_path} …")
    mat = load_dreamer_mat(mat_path)
    kw = dict(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        apply_filter=not args.no_filter,
        sfreq=args.sfreq,
    )
    n = count_dreamer_clips(mat, **kw)
    if n == 0:
        raise RuntimeError("No clips yielded — check MAT path and structure.")

    print(f"Allocating memmap for {n} clips, shape ({n}, 14, {args.chunk_size}) …")
    x_path = out_dir / "X.npy"
    X = np.lib.format.open_memmap(
        str(x_path),
        mode="w+",
        dtype=np.float32,
        shape=(n, 14, args.chunk_size),
    )
    sid = np.lib.format.open_memmap(
        str(out_dir / "subject_id.npy"), mode="w+", dtype=np.int32, shape=(n,)
    )
    tid = np.lib.format.open_memmap(
        str(out_dir / "trial_id.npy"), mode="w+", dtype=np.int32, shape=(n,)
    )
    st0 = np.lib.format.open_memmap(
        str(out_dir / "start_sample.npy"), mode="w+", dtype=np.int32, shape=(n,)
    )
    val = np.lib.format.open_memmap(
        str(out_dir / "valence.npy"), mode="w+", dtype=np.float32, shape=(n,)
    )
    aro = np.lib.format.open_memmap(
        str(out_dir / "arousal.npy"), mode="w+", dtype=np.float32, shape=(n,)
    )
    dom = np.lib.format.open_memmap(
        str(out_dir / "dominance.npy"), mode="w+", dtype=np.float32, shape=(n,)
    )

    for i, clip in enumerate(iter_dreamer_clips(mat, **kw)):
        if i % 500 == 0:
            print(f"  writing {i}/{n}")
        X[i] = clip.eeg
        sid[i] = clip.subject_id
        tid[i] = clip.trial_id
        st0[i] = clip.start_sample
        val[i] = clip.valence
        aro[i] = clip.arousal
        dom[i] = clip.dominance

    del X, sid, tid, st0, val, aro, dom

    manifest = {
        "version": 1,
        "dataset": "dreamer",
        "source_mat": str(mat_path.resolve()),
        "sfreq": args.sfreq,
        "ch_names": list(DREAMER_CHANNEL_NAMES),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "bandpass_hz": [1.0, 45.0] if not args.no_filter else None,
        "n_epochs": n,
        "x_file": "X.npy",
        "x_shape": [n, 14, args.chunk_size],
        "dtype": "float32",
        "labels": {
            "valence": "[1,5] Likert per trial (replicated per clip)",
            "arousal": "[1,5]",
            "dominance": "[1,5]",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Wrote {n} epochs to {out_dir}")
    print(json.dumps({k: manifest[k] for k in ("n_epochs", "x_shape", "chunk_size", "overlap")}, indent=2))


if __name__ == "__main__":
    main()
