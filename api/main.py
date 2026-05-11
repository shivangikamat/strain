"""FastAPI orchestrator for STRAIN."""

from __future__ import annotations

import io
import os
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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


DEMO_PATIENTS: list[dict[str, Any]] = [
    {
        "id": "alex-chen",
        "name": "Alex Chen",
        "avatar": "👨‍💼",
        "age": 32,
        "profession": "Software Engineer",
        "tag": "High Stress",
        "description": "Reports elevated anxiety and poor sleep over the past 3 months.",
        "epoch_index": 77160,
        "accent": "purple-yellow",
    },
    {
        "id": "maria-santos",
        "name": "Maria Santos",
        "avatar": "👩‍🎨",
        "age": 28,
        "profession": "Artist",
        "tag": "Calm & Focused",
        "description": "Meditative baseline. Strong alpha dominance and low arousal state.",
        "epoch_index": 1155,
        "accent": "cyan-purple",
    },
    {
        "id": "james-obrien",
        "name": "James O'Brien",
        "avatar": "👴",
        "age": 58,
        "profession": "Executive",
        "tag": "Elevated Arousal",
        "description": "High dominance, elevated beta activity. Active cognitive load detected.",
        "epoch_index": 33830,
        "accent": "yellow-orange",
    },
]


@api.get("/demo-patients")
def demo_patients() -> list[dict[str, Any]]:
    """Return the 3 pre-crafted demo patient profiles."""
    return DEMO_PATIENTS


# EMOTIV EPOC+ 14-channel positions (normalized, x=left-right, y=front-back)
_EPOC_POS: dict[str, tuple[float, float]] = {
    "AF3": (-0.22, 0.82), "F7": (-0.68, 0.58), "F3": (-0.38, 0.62),
    "FC5": (-0.63, 0.32), "T7": (-0.92, 0.0),  "P7": (-0.68, -0.58),
    "O1":  (-0.22, -0.88), "O2": (0.22, -0.88), "P8": (0.68, -0.58),
    "T8":  (0.92, 0.0),   "FC6": (0.63, 0.32), "F4":  (0.38, 0.62),
    "F8":  (0.68, 0.58),  "AF4": (0.22, 0.82),
}


@api.get("/brain-image/{epoch_index}")
def brain_image(epoch_index: int) -> Response:
    """Generate an EEG electrode activation topography PNG for a DREAMER epoch."""
    from strain.pipelines.dreamer_analyze import analyze_dreamer_epoch

    try:
        out = analyze_dreamer_epoch(epoch_index)
    except Exception:
        out = {}

    band_power: dict[str, float] = (out.get("features") or {}).get("band_mean_power") or {}

    # Pick beta band power per channel; fall back to a flat value if not available
    vals = {}
    for ch in _EPOC_POS:
        v = band_power.get(f"beta_{ch}") or band_power.get(f"theta_{ch}") or 0.5
        vals[ch] = float(v)

    all_v = list(vals.values()) or [0.5]
    vmin, vmax = min(all_v), max(all_v)
    if vmax == vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(5, 5.2), facecolor="#0d0b14")
    ax.set_facecolor("#0d0b14")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Scalp circle
    scalp = mpatches.Circle((0, 0), 1.0, color="#1e1b2e", zorder=1)
    ax.add_patch(scalp)
    scalp_ring = mpatches.Circle((0, 0), 1.0, color="#a855f7", fill=False, linewidth=1.5, zorder=2)
    ax.add_patch(scalp_ring)

    # Nose
    nose_x = [-.06, 0, .06]
    nose_y = [1.0, 1.18, 1.0]
    ax.plot(nose_x, nose_y, color="#a855f7", linewidth=1.5, zorder=3)

    # Left/right ear bumps
    for side in (-1, 1):
        ear = mpatches.Arc((side * 1.0, 0), 0.18, 0.3, angle=0,
                           theta1=90 if side < 0 else -90,
                           theta2=270 if side < 0 else 90,
                           color="#a855f7", linewidth=1.2, zorder=3)
        ax.add_patch(ear)

    # Colormap: purple → yellow (matches STRAIN palette)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "strain", ["#a855f7", "#22c55e", "#fbbf24"]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ch, (cx, cy) in _EPOC_POS.items():
        v = vals[ch]
        color = cmap(norm(v))
        glow = mpatches.Circle((cx, cy), 0.11, color=(*color[:3], 0.18), zorder=4)
        ax.add_patch(glow)
        dot = mpatches.Circle((cx, cy), 0.065, color=color, zorder=5)
        ax.add_patch(dot)
        ax.text(cx, cy - 0.17, ch, ha="center", va="top",
                fontsize=5.5, color="#a1a1aa", zorder=6, fontfamily="monospace")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cbar.ax.tick_params(colors="#71717a", labelsize=7)
    cbar.set_label("β power", color="#71717a", fontsize=7)
    cbar.outline.set_edgecolor("#3f3f46")

    ax.set_title("EEG Electrode Activation · EMOTIV EPOC+",
                 color="#a1a1aa", fontsize=8, pad=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


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
