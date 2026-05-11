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
import traceback
from typing import Any, Literal

import numpy as np
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
    build_dashboard_deeplink,
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
    "For Prompt Opinion BYO: call patient_screening_markdown_report_tool with PatientEmotionContext JSON; "
    "the tool returns JSON with a **markdown** string — render that field as the user-facing Markdown message "
    "(some MCP hosts require JSON tool output, not raw Markdown). "
    "Use example_patient_emotion_context_json_tool as a template. "
    "Set STRAIN_PUBLIC_DASHBOARD_URL to your public Vite/ngrok origin for dashboard links. "
    "MCP SSE/HTTP URL is only for tool transport — user-facing charts use the dashboard link inside markdown."
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
    try:
        if not caps.experimental:
            caps.experimental = {}
        caps.experimental["ai.promptopinion/fhir-context"] = {}
    except Exception:
        # Do not break initialize if capabilities are immutable or Po sends strict schema checks.
        pass
    return caps

mcp._mcp_server.get_capabilities = _get_capabilities_with_fhir


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _json(data: Any) -> str:
    """Serialize tool payloads; numpy / odd types must not break MCP JSON-RPC."""
    return json.dumps(data, indent=2, default=_json_default)


def _tool_fail(tool: str, exc: BaseException) -> str:
    return _json(
        {
            "error": "strain_mcp_tool_failed",
            "tool": tool,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "hint": "Check MCP server terminal; common causes: missing data/emotions.csv, "
            "missing strain/models/baseline_pipeline.joblib, or DREAMER export/VAD not trained.",
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    )


@mcp.tool()
def load_dataset_tool(
    dataset: str = "eeg_brainwave",
    csv_path: str | None = None,
    max_rows: int | None = None,
    processed_dir: str | None = None,
) -> str:
    """Dataset metadata: eeg_brainwave (Kaggle CSV) or dreamer (exported manifest)."""
    try:
        return _json(
            load_dataset_meta(
                dataset,
                csv_path=csv_path,
                max_rows=max_rows,
                processed_dir=processed_dir,
            )
        )
    except Exception as e:
        return _tool_fail("load_dataset_tool", e)


@mcp.tool()
def extract_features_tool(
    csv_path: str | None = None,
    row_index: int = 0,
) -> str:
    """Extract proxy band features for one row of the emotions CSV."""
    try:
        ds = load_emotions_csv(csv_path)
        idx = row_index % ds.X.shape[0]
        row = ds.X[idx]
        return _json(extract_features(row, ds.feature_names))
    except Exception as e:
        return _tool_fail("extract_features_tool", e)


@mcp.tool()
def extract_dreamer_epoch_features_tool(
    epoch_index: int = 0,
    processed_dir: str | None = None,
) -> str:
    """Welch band features for one DREAMER memmap epoch (requires export + manifest)."""
    try:
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
    except Exception as e:
        return _tool_fail("extract_dreamer_epoch_features_tool", e)


@mcp.tool()
def classify_emotion_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Classify emotion (NEGATIVE/NEUTRAL/POSITIVE) for one CSV row."""
    try:
        ds = load_emotions_csv(csv_path)
        idx = row_index % ds.X.shape[0]
        bundle = load_classifier_pipeline()
        return _json(classify_emotion(ds.X[idx], feature_names=ds.feature_names, bundle=bundle))
    except Exception as e:
        return _tool_fail("classify_emotion_tool", e)


@mcp.tool()
def predict_dreamer_vad_tool(epoch_index: int = 0, processed_dir: str | None = None) -> str:
    """Predict valence/arousal/dominance (1–5) from one DREAMER epoch (requires trained VAD joblib)."""
    try:
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
    except Exception as e:
        return _tool_fail("predict_dreamer_vad_tool", e)


@mcp.tool()
def explain_decision_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Linear-model attribution for the predicted class."""
    try:
        ds = load_emotions_csv(csv_path)
        idx = row_index % ds.X.shape[0]
        row = ds.X[idx]
        bundle = load_classifier_pipeline()
        pred = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
        return _json(explain_decision(row, ds.feature_names, pred, bundle=bundle))
    except Exception as e:
        return _tool_fail("explain_decision_tool", e)


@mcp.tool()
def screen_mental_health_tool(csv_path: str | None = None, row_index: int = 0) -> str:
    """Demo screening scores (not clinical)."""
    try:
        ds = load_emotions_csv(csv_path)
        idx = row_index % ds.X.shape[0]
        row = ds.X[idx]
        bundle = load_classifier_pipeline()
        feats = extract_features(row, ds.feature_names)
        cls = classify_emotion(row, feature_names=ds.feature_names, bundle=bundle)
        return _json(screen_mental_health(cls, feats))
    except Exception as e:
        return _tool_fail("screen_mental_health_tool", e)


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
    try:
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
    except Exception as e:
        return _tool_fail("export_fhir_tool", e)


@mcp.tool()
def example_patient_emotion_context_json_tool() -> str:
    """Example JSON object for a new demo patient (binds patient_id to CSV row 0). Use as a template in Po."""
    try:
        return _json(example_patient_context_dict())
    except Exception as e:
        return _tool_fail("example_patient_emotion_context_json_tool", e)


@mcp.tool()
def patient_screening_markdown_report_tool(
    patient_context_json: str,
    dashboard_base_url: str | None = None,
) -> str:
    """
    Run emotion + demo screening for one PatientEmotionContext.

    Returns JSON (Prompt Opinion-compatible) with a **markdown** field for the chat UI, plus optional **dashboard_url**.
    Pass dashboard_base_url or set STRAIN_PUBLIC_DASHBOARD_URL (e.g. https://your-vite.ngrok-free.app).
    """
    try:
        try:
            ctx = parse_patient_context_json(patient_context_json)
        except (json.JSONDecodeError, ValidationError) as e:
            return _json({"error": "invalid_patient_context_json", "detail": str(e)})
        base = (dashboard_base_url or os.environ.get("STRAIN_PUBLIC_DASHBOARD_URL") or "").strip() or None
        try:
            pipeline = run_emotion_pipeline_for_context(ctx)
        except FileNotFoundError as e:
            return _json({"error": "data_or_model_missing", "detail": str(e)})
        md = format_screening_markdown(ctx, pipeline, dashboard_base_url=base)
        out: dict[str, Any] = {
            "format": "markdown",
            "markdown": md,
            "hint": "Render the `markdown` field as the assistant reply (Markdown).",
        }
        if base:
            out["dashboard_url"] = build_dashboard_deeplink(base, ctx)
        return _json(out)
    except Exception as e:
        return _tool_fail("patient_screening_markdown_report_tool", e)


_DEMO_PATIENT_IDS: dict[str, str] = {
    "alex": "alex-chen",
    "alex chen": "alex-chen",
    "alex-chen": "alex-chen",
    "maria": "maria-santos",
    "maria santos": "maria-santos",
    "maria-santos": "maria-santos",
    "james": "james-obrien",
    "james o'brien": "james-obrien",
    "james obrien": "james-obrien",
    "james-obrien": "james-obrien",
}

_DEMO_PATIENT_META: dict[str, dict[str, str]] = {
    "alex-chen":   {"name": "Alex Chen",      "tag": "High Stress",       "epoch": "77160"},
    "maria-santos": {"name": "Maria Santos",  "tag": "Calm & Focused",    "epoch": "1155"},
    "james-obrien": {"name": "James O'Brien", "tag": "Elevated Arousal",  "epoch": "33830"},
}


@mcp.tool()
def get_demo_patient_dashboard_link_tool(
    patient_name: str,
    dashboard_base_url: str | None = None,
) -> str:
    """
    Return the dashboard deep-link URL for a named demo patient.

    Accepts first name, full name, or id slug:
      "Alex", "Alex Chen", "alex-chen" → https://…/?patient=alex-chen

    Set STRAIN_PUBLIC_DASHBOARD_URL or pass dashboard_base_url for the base URL.
    Returns JSON with: patient_id, name, tag, epoch_index, dashboard_url.
    """
    try:
        key = patient_name.strip().lower()
        pid = _DEMO_PATIENT_IDS.get(key)
        if not pid:
            available = ", ".join(m["name"] for m in _DEMO_PATIENT_META.values())
            return _json({"error": "unknown_patient", "available": available, "received": patient_name})

        base = (dashboard_base_url or os.environ.get("STRAIN_PUBLIC_DASHBOARD_URL") or "").rstrip("/")
        if not base:
            base = "http://localhost:5173"

        url = f"{base}/?patient={pid}"
        meta = _DEMO_PATIENT_META[pid]
        return _json({
            "patient_id": pid,
            "name": meta["name"],
            "tag": meta["tag"],
            "epoch_index": int(meta["epoch"]),
            "dashboard_url": url,
            "hint": f"Share this URL to open the full EEG analysis for {meta['name']} directly.",
        })
    except Exception as e:
        return _tool_fail("get_demo_patient_dashboard_link_tool", e)


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
