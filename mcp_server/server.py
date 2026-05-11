"""
STRAIN MCP server — ``strain-tools`` for Cursor, Prompt Opinion, or other MCP clients.

**Stdio (local):** ``python -m mcp_server.server`` or ``STRAIN_MCP_TRANSPORT=stdio``

**SSE (URL for Prompt Opinion + ngrok):**
``STRAIN_MCP_TRANSPORT=sse FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8765 STRAIN_MCP_RELAX_DNS=1 python -m mcp_server.server``

**Streamable HTTP (Prompt Opinion transport “Streamable HTTP”):**
``STRAIN_MCP_TRANSPORT=streamable-http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8765 STRAIN_MCP_RELAX_DNS=1 python -m mcp_server.server``
(or ``./scripts/run_mcp_streamable_http.sh``). Register ``https://<subdomain>.ngrok-free.app/mcp`` (default path).

See ``docs/mcp-setup-first-steps.md`` for matching Po transport ↔ URL path (``/sse`` vs ``/mcp``).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError

from strain.features.extract import extract_features
from strain.io.catalog import load_dataset_meta
from strain.io.emotions_csv import load_emotions_csv
from strain.models.classifier import (
    classify_emotion,
    explain_decision,
    load_classifier_pipeline,
)
from strain.io.fhir import generate_fhir_bundle
from strain.pipelines.dreamer_analyze import analyze_dreamer_epoch
from strain.screening.mental_health import screen_mental_health
from strain.demo.patient_context import (
    example_patient_context_dict,
    format_screening_markdown,
    parse_patient_context_json,
    run_emotion_pipeline_for_context,
)


def _sse_bind_from_env() -> tuple[str, int]:
    """FastMCP does not read FASTMCP_* from the environment unless passed into the constructor."""
    host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
    raw_port = os.environ.get("FASTMCP_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    return host, port


_SSE_HOST, _SSE_PORT = _sse_bind_from_env()

_MCP_INSTRUCTIONS = (
    "STRAIN v2.0 — healthcare hackathon prototype tools for EEG-derived emotion screening "
    "(tabular Kaggle features) and DREAMER multi-channel epochs (VAD regression). "
    "Outputs are for research and demonstration only — not a medical device. "
    "For Prompt Opinion BYO: use patient_screening_markdown_report_tool with JSON from "
    "example_patient_emotion_context_json_tool to return Markdown tables + a dashboard deep link "
    "(set STRAIN_PUBLIC_DASHBOARD_URL to your public Vite/ngrok origin). "
    "MCP SSE/HTTP URL is only for tool transport — user-facing charts use the dashboard link in Markdown."
)

_mcp_relax_dns = os.environ.get("STRAIN_MCP_RELAX_DNS", "").lower() in ("1", "true", "yes")

mcp = FastMCP(
    "strain-tools",
    instructions=_MCP_INSTRUCTIONS,
    host=_SSE_HOST,
    port=_SSE_PORT,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=not _mcp_relax_dns,
    ),
)


original_get_capabilities = mcp._mcp_server.get_capabilities

def _get_capabilities_with_fhir(*args: Any, **kwargs: Any) -> Any:
    caps = original_get_capabilities(*args, **kwargs)
    if not caps.experimental:
        caps.experimental = {}
    caps.experimental["ai.promptopinion/fhir-context"] = {}
    return caps

mcp._mcp_server.get_capabilities = _get_capabilities_with_fhir



def _json(data: Any) -> str:
    return json.dumps(data, indent=2)


@mcp.tool()
def load_dataset_tool(
    dataset: str = "eeg_brainwave",
    csv_path: str | None = None,
    max_rows: int | None = None,
    processed_dir: str | None = None,
) -> str:
    """Dataset metadata: eeg_brainwave (Kaggle CSV) or dreamer (exported manifest)."""
    return _json(
        load_dataset_meta(
            dataset,
            csv_path=csv_path,
            max_rows=max_rows,
            processed_dir=processed_dir,
        )
    )


@mcp.tool()
def extract_features_tool(
    csv_path: str | None = None,
    row_index: int = 0,
) -> str:
    """Extract proxy band features for one row of the emotions CSV."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    return _json(extract_features(row, ds.feature_names))


@mcp.tool()
def extract_dreamer_epoch_features_tool(
    epoch_index: int = 0,
    processed_dir: str | None = None,
) -> str:
    """Welch band features for one DREAMER memmap epoch (requires export + manifest)."""
    out = analyze_dreamer_epoch(epoch_index, processed_dir=processed_dir)
    return _json(
        {
            "epoch_index": out["epoch_index"],
            "subject_id": out["subject_id"],
            "trial_id": out["trial_id"],
            "true_vad": out["true_vad"],
            "features": out["features"],
        }
    )


@mcp.tool()
def classify_emotion_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Classify emotion (NEGATIVE/NEUTRAL/POSITIVE) for one CSV row."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    bundle = load_classifier_pipeline()
    return _json(classify_emotion(ds.X[idx], feature_names=ds.feature_names, bundle=bundle))


@mcp.tool()
def predict_dreamer_vad_tool(epoch_index: int = 0, processed_dir: str | None = None) -> str:
    """Predict valence/arousal/dominance (1–5) from one DREAMER epoch (requires trained VAD joblib)."""
    out = analyze_dreamer_epoch(epoch_index, processed_dir=processed_dir)
    return _json(
        {
            "epoch_index": out["epoch_index"],
            "true_vad": out["true_vad"],
            "predicted_vad": out["predicted_vad"],
            "explanation": out["explanation"],
            "screening": out["screening"],
        }
    )


@mcp.tool()
def explain_decision_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Linear-model attribution for the predicted class."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    bundle = load_classifier_pipeline()
    pred = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
    return _json(explain_decision(row, ds.feature_names, pred, bundle=bundle))


@mcp.tool()
def screen_mental_health_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Demo screening scores (not clinical)."""
    ds = load_emotions_csv(csv_path)
    idx = row_index % ds.X.shape[0]
    row = ds.X[idx]
    bundle = load_classifier_pipeline()
    feats = extract_features(row, ds.feature_names)
    cls = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
    return _json(screen_mental_health(cls, feats))


@mcp.tool()
def export_fhir_tool(
    ctx: Context,
    source: Literal["csv", "dreamer"] = "csv",
    index: int = 0,
    patient_id: str | None = None,
    csv_path: str | None = None,
    processed_dir: str | None = None,
) -> str:
    """Generate a FHIR R4 Bundle containing mental health risk assessments from EEG screening."""
    # 1. Grab FHIR context from Starlette headers if available
    req = getattr(ctx.request_context, "request", None)
    if req and hasattr(req, "headers"):
        if not patient_id:
            patient_id = req.headers.get("x-patient-id")

    # default fallback
    if not patient_id:
        patient_id = "demo-alex"

    if source == "dreamer":
        out = analyze_dreamer_epoch(index, processed_dir=processed_dir)
        screening = out.get("screening")
        if not screening:
            return _json({"error": "No screening result available (VAD model not trained?)"})
        return _json(generate_fhir_bundle(screening, patient_id=patient_id))
    else:
        ds = load_emotions_csv(csv_path)
        idx = index % ds.X.shape[0]
        row = ds.X[idx]
        bundle = load_classifier_pipeline()
        feats = extract_features(row, ds.feature_names)
        cls = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
        screening = screen_mental_health(cls, feats)
        return _json(generate_fhir_bundle(screening, patient_id=patient_id))


@mcp.tool()
def example_patient_emotion_context_json_tool() -> str:
    """Example JSON object for a new demo patient (binds patient_id to CSV row 0). Use as a template in Po."""
    return _json(example_patient_context_dict())


@mcp.tool()
def patient_screening_markdown_report_tool(
    patient_context_json: str,
    dashboard_base_url: str | None = None,
) -> str:
    """
    Run emotion + demo screening for one PatientEmotionContext; return Markdown (tables, link).

    Set dashboard_base_url or STRAIN_PUBLIC_DASHBOARD_URL (e.g. https://your-vite.ngrok-free.app)
    so the report includes a clickable **Launch STRAIN dashboard** link. This is separate from the MCP SSE URL.
    """
    try:
        ctx = parse_patient_context_json(patient_context_json)
    except (json.JSONDecodeError, ValidationError) as e:
        return _json({"error": "invalid_patient_context_json", "detail": str(e)})
    base = (dashboard_base_url or os.environ.get("STRAIN_PUBLIC_DASHBOARD_URL") or "").strip() or None
    try:
        pipeline = run_emotion_pipeline_for_context(ctx)
    except FileNotFoundError as e:
        return _json({"error": "data_or_model_missing", "detail": str(e)})
    return format_screening_markdown(ctx, pipeline, dashboard_base_url=base)


def main() -> None:
    parser = argparse.ArgumentParser(description="STRAIN MCP Server")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run over SSE on 127.0.0.1 (Prompt Opinion local); use --port (default 8001).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port when using --sse (ignored for stdio / env-driven transports).",
    )
    args = parser.parse_args()

    if args.sse:
        # FastMCP.run() does not accept host/port; mutate settings before serving.
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = args.port
        print(
            "Starting STRAIN FastMCP Server in SSE mode "
            f"({mcp.settings.host}:{mcp.settings.port})…"
        )
        mcp.run(transport="sse")
        return

    transport = os.environ.get("STRAIN_MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "sse", "streamable-http"):
        raise SystemExit(
            f"Invalid STRAIN_MCP_TRANSPORT={transport!r}. "
            "Use stdio, sse, or streamable-http."
        )
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
