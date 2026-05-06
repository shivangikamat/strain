#!/usr/bin/env python3
"""Train LogisticRegression baseline and save to strain/models/baseline_pipeline.joblib."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain.models.classifier import train_and_save_baseline  # noqa: E402


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    out = train_and_save_baseline(csv)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
