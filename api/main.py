"""FastAPI orchestrator for EmotiScan."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from emotiscan.agents.analysis_engine import AnalysisEngine
from emotiscan.agents.orchestrator import Orchestrator
from emotiscan.io.catalog import load_dataset_meta
from emotiscan.io.emotions_csv import load_dataset, load_emotions_csv
from emotiscan.models.classifier import train_and_save_baseline
from emotiscan.models.dreamer_vad import train_and_save_dreamer_vad

app = FastAPI(title="EmotiScan API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orch = Orchestrator()

api = APIRouter(prefix="/api")


class AgentRunRequest(BaseModel):
    query: str = Field(
        ...,
        description='Use "row=123" or "epoch=456". When source=dreamer, epoch index is used.',
    )
    source: Literal["csv", "dreamer"] = "csv"
    csv_path: str | None = None
    dreamer_processed_dir: str | None = None


class AnalyzeRequest(BaseModel):
    row_index: int = 0
    csv_path: str | None = None


class AnalyzeDreamerRequest(BaseModel):
    epoch_index: int = 0
    processed_dir: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/agent/run")
def agent_run(body: AgentRunRequest) -> dict[str, Any]:
    try:
        return _orch.run(
            body.query,
            source=body.source,
            csv_path=body.csv_path,
            dreamer_processed_dir=body.dreamer_processed_dir,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@api.post("/analyze")
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


@api.post("/analyze/dreamer")
def analyze_dreamer(body: AnalyzeDreamerRequest) -> dict[str, Any]:
    try:
        from emotiscan.pipelines.dreamer_analyze import analyze_dreamer_epoch

        return analyze_dreamer_epoch(
            body.epoch_index,
            processed_dir=body.processed_dir,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@api.get("/dataset/meta")
def dataset_meta(csv_path: str | None = None) -> dict[str, Any]:
    try:
        return load_dataset("eeg_brainwave", csv_path=csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@api.get("/dataset/dreamer/meta")
def dreamer_processed_meta(processed_dir: str | None = None) -> dict[str, Any]:
    """Metadata for exported DREAMER epoch tensors (after running export script)."""
    try:
        return load_dataset_meta("dreamer", processed_dir=processed_dir)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e


@api.post("/internal/train-baseline")
def train_baseline(csv_path: str | None = None) -> dict[str, Any]:
    """Train and save sklearn baseline (dev convenience)."""
    return train_and_save_baseline(csv_path)


@api.post("/internal/train-dreamer-vad")
def train_dreamer_vad_endpoint(
    processed_dir: str | None = Query(default=None),
    test_size: float = Query(default=0.2, ge=0.05, le=0.5),
    max_train_samples: int | None = Query(default=None),
) -> dict[str, Any]:
    """Train subject-holdout VAD regressor on exported epochs."""
    from pathlib import Path

    pd = Path(processed_dir) if processed_dir else None
    return train_and_save_dreamer_vad(
        processed_dir=pd,
        test_size=test_size,
        max_train_samples=max_train_samples,
    )


app.include_router(api)
