# Strain / EmotiScan v2.0

Hackathon prototype: emotion classification and **non-clinical** demo screening from Kaggle-style tabular EEG features in [`data/emotions.csv`](data/emotions.csv) (default path; override with `EMOTISCAN_EMOTIONS_CSV`).

## DREAMER — real multi-channel EEG epoch tensors (recommended next step)

**Why DREAMER:** single open file ([Zenodo record 546113](https://zenodo.org/records/546113)), **14 channels @ 128 Hz**, trial-level **valence / arousal / dominance** (1–5). No access request (unlike DEAP/SEED), so it is the most **accessible** path to true brainwave tensors that still matches the hackathon plan (low-channel validation + biomarker story).

**Optimal epoch strategy for this repo:** sliding windows **256 samples (~2 s) with 50% overlap (128)** → fixed-shape tensors `(14, 256)` ideal for PyTorch `Dataset` / future 4D-CRNN-style models, plus more training rows per subject than “one vector per trial”. Labels are copied from the parent trial (standard for clip-level learning on DREAMER). Optional **1–45 Hz** bandpass via MNE during export.

1. Download `DREAMER.mat` into `data/raw/DREAMER.mat` (or set `EMOTISCAN_DREAMER_MAT`).
2. Export memmapped tensors + `manifest.json`:

```bash
source .venv/bin/activate
python scripts/export_dreamer_epochs.py --mat data/raw/DREAMER.mat
```

3. Read from Python:

```python
from emotiscan.data.dreamer_epochs import load_dreamer_manifest, open_dreamer_X_memmap
from emotiscan.features.eeg_epoch import extract_features_from_epoch

meta = load_dreamer_manifest()
X = open_dreamer_X_memmap()  # mmap (n, 14, 256)
feats = extract_features_from_epoch(X[0], sfreq=meta["sfreq"])
```

4. After export, `GET http://127.0.0.1:8000/dataset/dreamer/meta` returns the manifest JSON.

5. **Subject-holdout VAD model** (Ridge on Welch + channel-variance features):

```bash
python scripts/train_dreamer_vad.py
# optional smoke: --max-train-samples 4000
```

This writes `emotiscan/models/dreamer_vad_multiridge.joblib`. Then `POST /api/analyze/dreamer` and the UI “DREAMER epochs” tab return predictions + explanations.

6. **PyTorch `Dataset`**: `from emotiscan.data import DreamerEpochDataset` — optional `indices` from `train_test_mask_by_subject()` for subject-safe splits.

7. **Agent / API routing**: `POST /api/agent/run` with body `{"query": "epoch=42", "source": "dreamer"}` runs the DREAMER pipeline (optional `dreamer_processed_dir`).

Outputs live under `data/processed/dreamer/` (ignored with `data/` in `.gitignore`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Train the baseline classifier (writes `emotiscan/models/baseline_pipeline.joblib`):

```bash
python scripts/train_baseline.py
```

EDA script:

```bash
python -m emotiscan.eda.kaggle_brainwave
```

## API + UI

Terminal 1 — FastAPI (`/agent/run`, `/analyze`, `/dataset/meta`):

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — Vite (proxies `/api` → FastAPI):

```bash
cd backend/frontend && npm run dev
```

Open `http://localhost:5173` and pick a row index to inspect predictions and demo risk scores.

## MCP server (`emotiscan-tools`)

**First-time setup (venv, data, Cursor, optional SSE/ngrok):** [docs/mcp-setup-first-steps.md](docs/mcp-setup-first-steps.md).

Quick local stdio:

```bash
source .venv/bin/activate
python -m mcp_server.server
```

Or: `./scripts/run_mcp_stdio.sh`. This repo includes [`.cursor/mcp.json`](.cursor/mcp.json) so Cursor can load **emotiscan-tools** when you open the folder (uses `.venv/bin/python`; create the venv first or edit the path).

SSE for Prompt Opinion + ngrok: `./scripts/run_mcp_sse.sh` — see the first-steps doc. Env template: [`.env.example`](.env.example).

**Prompt Opinion / “Agents Assemble” hackathon:** [docs/prompt-opinion-hackathon.md](docs/prompt-opinion-hackathon.md) (repos, ngrok URL, A2A via `po-adk-python`).

## Layout

- `emotiscan/` — loaders, features, sklearn model, screening stub, in-process agents
- `emotiscan/io/dreamer_mat.py` — DREAMER.mat reader + sliding-window clip iterator
- `emotiscan/data/dreamer_epochs.py` — mmap manifest helpers
- `scripts/export_dreamer_epochs.py` — MAT → `data/processed/dreamer/` tensors
- `api/` — FastAPI orchestrator
- `mcp_server/` — MCP tools (stdio default; SSE for remote)
- `.cursor/mcp.json` — Cursor project MCP entry for **emotiscan-tools**
- `docs/mcp-setup-first-steps.md` — MCP setup checklist
- `scripts/` — Kaggle download helper, training CLI
- `notebooks/01_kaggle_eeg_eda.ipynb` — exploratory notebook
- `backend/frontend/` — React + Recharts dashboard
