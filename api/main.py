"""FastAPI orchestrator for STRAIN."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from strain.agents.analysis_engine import AnalysisEngine
from strain.agents.orchestrator import Orchestrator
from strain.io.catalog import load_dataset_meta
from strain.io.emotions_csv import load_dataset, load_emotions_csv
from strain.models.classifier import train_and_save_baseline
from strain.models.dreamer_vad import train_and_save_dreamer_vad
from strain.demo.patient_context import (
    PatientEmotionContext,
    build_dashboard_deeplink,
    format_screening_markdown,
    run_emotion_pipeline_for_context,
)

app = FastAPI(title="STRAIN API", version="0.1.0")

_cors_base = ["http://localhost:5173", "http://127.0.0.1:5173"]
_cors_extra = os.environ.get("STRAIN_CORS_ORIGINS", "").strip()
if _cors_extra:
    _cors_base = [
        *_cors_base,
        *[o.strip() for o in _cors_extra.split(",") if o.strip()],
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_base,
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


class ExportFhirRequest(BaseModel):
    source: Literal["csv", "dreamer"] = "csv"
    index: int = 0
    patient_id: str = "demo-alex"
    csv_path: str | None = None
    dreamer_processed_dir: str | None = None


class PatientSummaryRequest(BaseModel):
    """Run STRAIN for one demo patient and return Markdown + dashboard deep link."""

    patient: PatientEmotionContext
    dashboard_base_url: str | None = Field(
        default=None,
        description="Public origin of the Vite app, e.g. https://xxx.ngrok-free.app (defaults to STRAIN_PUBLIC_DASHBOARD_URL).",
    )


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
        from strain.pipelines.dreamer_analyze import analyze_dreamer_epoch

        return analyze_dreamer_epoch(
            body.epoch_index,
            processed_dir=body.processed_dir,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@api.post("/patient/summary")
def patient_summary(body: PatientSummaryRequest) -> dict[str, Any]:
    """Validate patient context JSON, run analysis, return Markdown + deep link for the web UI."""
    base = (body.dashboard_base_url or os.environ.get("STRAIN_PUBLIC_DASHBOARD_URL") or "").strip()
    try:
        pipeline = run_emotion_pipeline_for_context(body.patient)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    out: dict[str, Any] = {
        "patient": body.patient.model_dump(),
        "pipeline": pipeline,
        "markdown": format_screening_markdown(
            body.patient,
            pipeline,
            dashboard_base_url=base or None,
        ),
    }
    if base:
        out["dashboard_url"] = build_dashboard_deeplink(base, body.patient)
    return out


@api.post("/export/fhir")
def export_fhir(body: ExportFhirRequest) -> dict[str, Any]:
    from strain.io.fhir import generate_fhir_bundle
    try:
        if body.source == "dreamer":
            from strain.pipelines.dreamer_analyze import analyze_dreamer_epoch
            out = analyze_dreamer_epoch(body.index, processed_dir=body.dreamer_processed_dir)
            screening = out.get("screening")
            if not screening:
                raise HTTPException(status_code=400, detail="No screening result (VAD model not trained?)")
            return generate_fhir_bundle(screening, patient_id=body.patient_id)
        else:
            from strain.features.extract import extract_features
            from strain.models.classifier import classify_emotion, load_classifier_pipeline
            from strain.screening.mental_health import screen_mental_health
            ds = load_emotions_csv(body.csv_path)
            idx = body.index % ds.X.shape[0]
            row = ds.X[idx]
            bundle = load_classifier_pipeline()
            feats = extract_features(row, ds.feature_names)
            cls = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
            screening = screen_mental_health(cls, feats)
            return generate_fhir_bundle(screening, patient_id=body.patient_id)
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
