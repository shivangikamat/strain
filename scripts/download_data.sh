#!/usr/bin/env bash
# Download Kaggle EEG brainwave dataset (requires ~/.kaggle/kaggle.json).
# Default hackathon path: unzip to data/raw/eeg_brainwave/
# This repo uses data/emotions.csv — copy or symlink the CSV there after unzip.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Optional: Kaggle download (birdy654/eeg-brainwave-dataset-feeling-emotions)"
if command -v kaggle &>/dev/null; then
  mkdir -p data/raw/eeg_brainwave
  kaggle datasets download -d birdy654/eeg-brainwave-dataset-feeling-emotions -p data/raw/
  unzip -o data/raw/eeg-brainwave-dataset-feeling-emotions.zip -d data/raw/eeg_brainwave/
  echo "Unzipped under data/raw/eeg_brainwave/. Copy the emotions CSV to data/emotions.csv if needed."
else
  echo "kaggle CLI not installed. pip install kaggle && set up ~/.kaggle/kaggle.json"
fi

if [[ -f data/emotions.csv ]]; then
  echo "Found data/emotions.csv (EmotiScan default — set EMOTISCAN_EMOTIONS_CSV to override)."
  python3 -c "import pandas as pd; df=pd.read_csv('data/emotions.csv', nrows=0); print('columns', len(df.columns))"
else
  echo "No data/emotions.csv yet — add the Kaggle export as data/emotions.csv"
fi
