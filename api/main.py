"""FastAPI orchestrator for EmotiScan."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from emotiscan.agents.analysis_engine import AnalysisEngine
from emotiscan.agents.orchestrator import Orchestrator
from emotiscan.io.emotions_csv import load_dataset, load_emotions_csv
from emotiscan.models.classifier import train_and_save_baseline

app = FastAPI(title="EmotiScan API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orch = Orchestrator()


class AgentRunRequest(BaseModel):
    query: str = Field(..., description='Use optional "row=123" in text to pick a row index.')
    csv_path: str | None = None


class AnalyzeRequest(BaseModel):
    row_index: int = 0
    csv_path: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/run")
def agent_run(body: AgentRunRequest) -> dict[str, Any]:
    try:
        return _orch.run(body.query, csv_path=body.csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/analyze")
def analyze(body: AnalyzeRequest) -> dict[str, Any]:
    try:
        ds = load_emotions_csv(body.csv_path)
        idx = body.row_index % ds.X.shape[0]
        eng = AnalysisEngine()
        row = ds.X[idx].tolist()
        analysis = eng.analyze_row(row, ds.feature_names, with_explanation=True)
        return {
            "row_index": idx,
            "ground_truth_label": str(ds.y[idx]),
            "analysis": analysis,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/dataset/meta")
def dataset_meta(csv_path: str | None = None) -> dict[str, Any]:
    try:
        return load_dataset("eeg_brainwave", csv_path=csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/dataset/dreamer/meta")
def dreamer_processed_meta() -> dict[str, Any]:
    """Metadata for exported DREAMER epoch tensors (after running export script)."""
    try:
        from emotiscan.data.dreamer_epochs import load_dreamer_manifest

        return load_dreamer_manifest()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e


@app.post("/internal/train-baseline")
def train_baseline(csv_path: str | None = None) -> dict[str, Any]:
    """Train and save sklearn baseline (dev convenience)."""
    return train_and_save_baseline(csv_path)
