#!/usr/bin/env python3
"""Train subject-holdout VAD regressor on exported DREAMER epochs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotiscan.models.dreamer_vad import train_and_save_dreamer_vad  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory with manifest + X.npy (default: data/processed/dreamer)",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for faster smoke training",
    )
    args = ap.parse_args()
    out = train_and_save_dreamer_vad(
        processed_dir=args.processed_dir,
        test_size=args.test_size,
        max_train_samples=args.max_train_samples,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
