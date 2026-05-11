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
from pathlib import Path
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


_REC_CLINICAL: dict[str, str] = {
    "no_concern": "No elevated neural markers in this scan window. Routine follow-up recommended.",
    "monitor": "Subclinical markers detected. Recommend repeat screening in 4–6 weeks.",
    "consult_pcp": "Elevated markers observed. Recommend consultation with a primary care provider.",
    "seek_specialist": "Significant neural stress markers. Recommend specialist referral for further evaluation.",
}

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
    "sam": "sam-rivera",
    "sam rivera": "sam-rivera",
    "sam-rivera": "sam-rivera",
}

_DEMO_PATIENT_META: dict[str, dict[str, str]] = {
    "alex-chen":    {"name": "Alex Chen",      "tag": "High Stress",        "epoch": "77160"},
    "maria-santos": {"name": "Maria Santos",   "tag": "Calm & Focused",     "epoch": "1155"},
    "james-obrien": {"name": "James O'Brien",  "tag": "Elevated Arousal",   "epoch": "33830"},
    "sam-rivera":   {"name": "Sam Rivera",     "tag": "Cognitive Overload",  "epoch": "12600"},
}


def _quickchart_url(cfg: dict) -> str:
    """Encode a QuickChart config as a URL (public, no API key needed)."""
    import urllib.parse, json as _json_mod
    return "https://quickchart.io/chart?w=500&h=220&bkg=%230d0b14&c=" + urllib.parse.quote(
        _json_mod.dumps(cfg, separators=(",", ":"))
    )


def _recommendations(
    dep: float, anx: float, cog: float,
    v_pred: float | str, a_pred: float | str, emotion: str,
) -> list[str]:
    steps: list[str] = []
    if isinstance(v_pred, float) and v_pred <= 2.5:
        steps.append("**Behavioral activation:** Schedule two enjoyable activities per day to counter low-valence patterns.")
    if isinstance(a_pred, float) and a_pred >= 3.5:
        steps.append("**Arousal regulation:** Practice 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s) before high-demand tasks.")
    if dep > 65:
        steps.append("**Depression markers elevated:** Initiate PHQ-9 structured assessment. Consider MBSR referral.")
    elif dep > 40:
        steps.append("**Subclinical depression signals:** Weekly mood journaling + 30 min aerobic activity recommended.")
    if anx > 65:
        steps.append("**Anxiety markers elevated:** GAD-7 follow-up indicated. Progressive muscle relaxation exercises daily.")
    elif anx > 40:
        steps.append("**Mild anxiety signals:** Limit caffeine after noon. Introduce 10-min daily mindfulness practice.")
    if cog >= 95:
        steps.append("**Extreme cognitive overload (β/α > 50):** Immediate task offload recommended. Stop multitasking now. 20-minute low-stimulation break required before resuming work. ADHD screening (Conners Adult ADHD Rating Scale) indicated given sustained beta elevation pattern.")
    elif cog > 65:
        steps.append("**High cognitive load:** Enforce 90-minute focused work cycles with mandatory 20-minute recovery. Avoid multitasking.")
    elif cog > 40:
        steps.append("**Moderate cognitive load:** Consider task-batching and single-session focus protocols.")
    if emotion == "POSITIVE" and dep <= 40 and anx <= 40:
        steps.append("**Positive baseline detected:** Current stress-management strategies appear effective. Maintain sleep schedule and exercise routine.")
    if not steps:
        steps.append("No significant neural stress markers in this scan window. Maintain current wellness practices.")
    return steps


@mcp.tool()
def analyze_named_patient_tool(
    patient_name: str,
    dashboard_base_url: str | None = None,
) -> str:
    """
    Run a full STRAIN EEG neural screening for a named enrolled patient.

    Accepts: "Alex Chen", "Maria Santos", "James O'Brien" (or first name / slug).
    Returns JSON with a **markdown** field — the full clinical report with brain scan image,
    risk charts, VAD charts, model interpretation, and personalised recommended steps.

    Set STRAIN_PUBLIC_DASHBOARD_URL or pass dashboard_base_url for image and dashboard URLs.
    """
    try:
        import datetime
        key = patient_name.strip().lower()
        pid = _DEMO_PATIENT_IDS.get(key)
        if not pid:
            available = ", ".join(m["name"] for m in _DEMO_PATIENT_META.values())
            return _json({"error": "unknown_patient", "available": available, "received": patient_name})

        meta = _DEMO_PATIENT_META[pid]
        epoch_idx = int(meta["epoch"])

        out = analyze_dreamer_epoch(epoch_idx)
        pred = out.get("predicted_vad") or {}
        screen = out.get("screening") or {}
        expl = out.get("explanation") or {}
        true_vad = out.get("true_vad") or {}
        features = out.get("features") or {}
        ratios = features.get("spectral_ratios") or {}

        base = (dashboard_base_url or os.environ.get("STRAIN_PUBLIC_DASHBOARD_URL") or "").rstrip("/")
        if not base:
            base = "http://localhost:5173"
        dashboard_url = f"{base}/?patient={pid}"
        brain_img_url = f"{base}/api/brain-image/{epoch_idx}"

        dep = float(screen.get("depression_risk", {}).get("score", 0))
        anx = float(screen.get("anxiety_risk", {}).get("score", 0))
        cog = float(screen.get("cognitive_load", {}).get("score", 0))
        rec = screen.get("recommendation", "no_concern")
        nl = expl.get("natural_language_explanation", "")

        v_pred = pred.get("valence", "N/A")
        a_pred = pred.get("arousal", "N/A")
        d_pred = pred.get("dominance", "N/A")
        v_true = true_vad.get("valence", "N/A")
        a_true = true_vad.get("arousal", "N/A")
        d_true = true_vad.get("dominance", "N/A")
        beta_alpha = ratios.get("beta_alpha", 0.0)
        theta_alpha = ratios.get("theta_alpha", 0.0)

        emotion = "POSITIVE" if isinstance(v_pred, float) and v_pred >= 3.5 else \
                  "NEGATIVE" if isinstance(v_pred, float) and v_pred <= 2.5 else "NEUTRAL"

        top_feats = []
        for feats_list in expl.get("per_target_top_features", {}).values():
            top_feats.extend(feats_list)
        seen: set[str] = set()
        deduped = []
        for f in sorted(top_feats, key=lambda x: abs(x.get("contribution", 0)), reverse=True):
            if f.get("name") not in seen:
                seen.add(f["name"])
                deduped.append(f)
        deduped = deduped[:5]

        # QuickChart — risk bar chart
        risk_chart = _quickchart_url({
            "type": "horizontalBar",
            "data": {
                "labels": ["Depression", "Anxiety", "Cognitive Load"],
                "datasets": [{
                    "data": [round(dep, 1), round(anx, 1), round(cog, 1)],
                    "backgroundColor": ["rgba(168,85,247,0.85)", "rgba(251,191,36,0.85)", "rgba(6,182,212,0.85)"],
                    "borderColor": ["#a855f7", "#fbbf24", "#06b6d4"],
                    "borderWidth": 2,
                }],
            },
            "options": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Neural Risk Indicators (/100)", "fontColor": "#a1a1aa"},
                "scales": {
                    "xAxes": [{"ticks": {"min": 0, "max": 100, "fontColor": "#71717a"}, "gridLines": {"color": "#27272a"}}],
                    "yAxes": [{"ticks": {"fontColor": "#a1a1aa"}, "gridLines": {"display": False}}],
                },
            },
        })

        # QuickChart — VAD grouped bar
        vad_labels = ["Valence", "Arousal", "Dominance"]
        pred_vals = [
            round(v_pred, 2) if isinstance(v_pred, float) else 0,
            round(a_pred, 2) if isinstance(a_pred, float) else 0,
            round(d_pred, 2) if isinstance(d_pred, float) else 0,
        ]
        true_vals = [
            round(v_true, 2) if isinstance(v_true, float) else 0,
            round(a_true, 2) if isinstance(a_true, float) else 0,
            round(d_true, 2) if isinstance(d_true, float) else 0,
        ]
        vad_chart = _quickchart_url({
            "type": "bar",
            "data": {
                "labels": vad_labels,
                "datasets": [
                    {"label": "Predicted", "data": pred_vals,
                     "backgroundColor": "rgba(168,85,247,0.85)", "borderColor": "#a855f7", "borderWidth": 1},
                    {"label": "Reference", "data": true_vals,
                     "backgroundColor": "rgba(34,211,238,0.55)", "borderColor": "#22d3ee", "borderWidth": 1},
                ],
            },
            "options": {
                "title": {"display": True, "text": "Affective State — Predicted vs Reference (1–5)", "fontColor": "#a1a1aa"},
                "scales": {
                    "yAxes": [{"ticks": {"min": 0, "max": 5, "fontColor": "#71717a"}, "gridLines": {"color": "#27272a"}}],
                    "xAxes": [{"ticks": {"fontColor": "#a1a1aa"}, "gridLines": {"display": False}}],
                },
                "legend": {"labels": {"fontColor": "#a1a1aa"}},
            },
        })

        recs = _recommendations(dep, anx, cog, v_pred, a_pred, emotion)

        md_lines = [
            f"## Neural EEG Screening — {meta['name']}",
            "",
            f"| | |",
            f"| --- | --- |",
            f"| **Patient** | {meta['name']} |",
            f"| **Profile** | {meta['tag']} |",
            f"| **EEG** | 14-ch EMOTIV EPOC+ · 128 Hz · 2 s epoch · Welch PSD |",
            f"| **Affective state** | **{emotion}** · β/α {beta_alpha:.3f} · θ/α {theta_alpha:.3f} |",
            f"| **Scan** | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} UTC · Epoch {epoch_idx} |",
            "",
            "### EEG Electrode Activation Map",
            "",
            f"![EEG Brain Scan — {meta['name']}]({brain_img_url})",
            "",
            "### Valence · Arousal · Dominance",
            "",
            f"![VAD Chart]({vad_chart})",
            "",
            "| Dimension | Predicted | Reference |",
            "| --- | ---: | ---: |",
            f"| Valence | **{v_pred:.2f}** | {v_true:.2f} |" if isinstance(v_pred, float) else "| Valence | N/A | — |",
            f"| Arousal | **{a_pred:.2f}** | {a_true:.2f} |" if isinstance(a_pred, float) else "| Arousal | N/A | — |",
            f"| Dominance | **{d_pred:.2f}** | {d_true:.2f} |" if isinstance(d_pred, float) else "| Dominance | N/A | — |",
            "",
            "### Neural Risk Indicators",
            "",
            f"![Risk Chart]({risk_chart})",
            "",
            "| Indicator | Score | Status |",
            "| --- | ---: | --- |",
            f"| Depression markers | **{dep:.1f}** / 100 | {'⚠ Elevated' if dep > 65 else '✓ Normal'} |",
            f"| Anxiety markers | **{anx:.1f}** / 100 | {'⚠ Elevated' if anx > 65 else '✓ Normal'} |",
            f"| Cognitive load | **{cog:.1f}** / 100 | {'⚠ Elevated' if cog > 65 else '✓ Normal'} |",
            "",
            f"**Assessment:** {_REC_CLINICAL.get(rec, rec)}",
            "",
        ]

        if nl:
            md_lines += ["### EEG Model Interpretation", "", nl, ""]

        if deduped:
            md_lines += ["### Top Signal Contributions", ""]
            for f in deduped:
                sign = "+" if f.get("contribution", 0) >= 0 else ""
                md_lines.append(f"- `{f['name']}` → {sign}{f.get('contribution', 0):.4f}")
            md_lines.append("")

        md_lines += ["### Recommended Next Steps", ""]
        for i, step in enumerate(recs, 1):
            md_lines.append(f"{i}. {step}")
        md_lines.append("")

        report_url = f"{base}/api/report/{pid}"
        md_lines += [
            "---",
            "",
            f"[**Open interactive STRAIN dashboard →**]({dashboard_url})",
            "",
            f"[**Download / view full report (Markdown) →**]({report_url})",
            "",
            "> *STRAIN EEG screening is for research purposes. Not a medical device. Not for clinical diagnosis or treatment decisions.*",
        ]

        md = "\n".join(md_lines)

        # Save report to disk so FastAPI can serve it as a viewable/downloadable file
        _reports_dir = Path(__file__).parent.parent / "data" / "reports"
        _reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = _reports_dir / f"{pid}.md"
        report_path.write_text(md, encoding="utf-8")

        return _json({
            "format": "markdown",
            "markdown": md,
            "dashboard_url": dashboard_url,
            "report_url": report_url,
            "patient_id": pid,
            "name": meta["name"],
            "epoch_index": epoch_idx,
        })
    except Exception as e:
        return _tool_fail("analyze_named_patient_tool", e)


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