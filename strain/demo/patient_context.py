"""
Patient ↔ STRAIN analysis binding for demos (Prompt Opinion BYO, MCP, deep links).

Maps a *synthetic* patient record to either a Kaggle CSV row or a DREAMER epoch index.
Not clinical — ties UI / agents to reproducible rows in research data.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from strain.agents.analysis_engine import AnalysisEngine
from strain.io.emotions_csv import load_emotions_csv
from strain.pipelines.dreamer_analyze import analyze_dreamer_epoch

_REC_COPY: dict[str, str] = {
    "no_concern": "No elevated demo scores from this toy model.",
    "monitor": "Demo-only: consider routine wellness habits; not clinical advice.",
    "consult_pcp": "Demo text only — discuss concerns with a primary care clinician in real life.",
    "seek_specialist": "Demo escalation language only — not a referral.",
}


class PatientEmotionContext(BaseModel):
    """JSON you pass to MCP / agents to run the emotion + screening stack for one demo patient."""

    patient_id: str = Field(
        ...,
        description="Stable id (FHIR Patient logical id style), e.g. pat-demo-0001",
    )
    display_name: str | None = Field(
        None,
        description="Shown in Markdown / UI banner when deep-linking",
    )
    source: Literal["csv", "dreamer"] = Field(
        "csv",
        description="csv = Kaggle tabular row; dreamer = exported epoch tensor",
    )
    row_index: int = Field(0, ge=0, description="CSV row when source=csv")
    epoch_index: int = Field(0, ge=0, description="Epoch index when source=dreamer")
    csv_path: str | None = None
    dreamer_processed_dir: str | None = None
    cohort: str | None = Field(None, description="Optional cohort label for reports")


def example_patient_context_dict() -> dict[str, Any]:
    """Example JSON object for a new demo patient (CSV row 0)."""
    return PatientEmotionContext(
        patient_id="pat-demo-0001",
        display_name="Demo Patient A",
        source="csv",
        row_index=0,
        epoch_index=0,
        cohort="prompt-opinion-launchpad",
    ).model_dump()


def parse_patient_context_json(raw: str) -> PatientEmotionContext:
    data = json.loads(raw)
    return PatientEmotionContext.model_validate(data)


def build_dashboard_deeplink(
    base_url: str,
    ctx: PatientEmotionContext,
    *,
    extra_query: dict[str, str] | None = None,
) -> str:
    """
    Build an HTTPS URL to the Vite dashboard with query params it understands.

    ``base_url`` should be the public origin only, e.g. ``https://abc.ngrok-free.app``
    or ``http://127.0.0.1:5173`` (no trailing path).
    """
    root = base_url.rstrip("/")
    q: dict[str, str] = {
        "mode": ctx.source,
        "patientId": ctx.patient_id,
    }
    if ctx.display_name:
        q["patientName"] = ctx.display_name
    if ctx.source == "csv":
        q["row"] = str(ctx.row_index)
    else:
        q["epoch"] = str(ctx.epoch_index)
    if extra_query:
        q.update(extra_query)
    return f"{root}/?{urlencode(q)}"


def run_emotion_pipeline_for_context(ctx: PatientEmotionContext) -> dict[str, Any]:
    """Run the same analyses the FastAPI ``/api/analyze`` stack uses, return JSON-serializable dict."""
    if ctx.source == "csv":
        ds = load_emotions_csv(ctx.csv_path)
        idx = ctx.row_index % ds.X.shape[0]
        eng = AnalysisEngine()
        row = ds.X[idx].tolist()
        analysis = eng.analyze_row(row, ds.feature_names, with_explanation=True)
        return {
            "patient": ctx.model_dump(),
            "source": "csv",
            "row_index": idx,
            "analysis": analysis,
        }
    out = analyze_dreamer_epoch(
        ctx.epoch_index,
        processed_dir=ctx.dreamer_processed_dir,
    )
    return {
        "patient": ctx.model_dump(),
        "source": "dreamer",
        "epoch_index": out.get("epoch_index"),
        "subject_id": out.get("subject_id"),
        "trial_id": out.get("trial_id"),
        "true_vad": out.get("true_vad"),
        "features": out.get("features"),
        "predicted_vad": out.get("predicted_vad"),
        "explanation": out.get("explanation"),
        "screening": out.get("screening"),
    }


def format_screening_markdown(
    ctx: PatientEmotionContext,
    pipeline: dict[str, Any],
    *,
    dashboard_base_url: str | None,
) -> str:
    """Rich Markdown for Prompt Opinion / BYO chats: tables, blockquotes, dashboard link."""
    name = ctx.display_name or ctx.patient_id
    lines: list[str] = [
        f"## STRAIN — demo screening · **{name}**",
        "",
        f"- **Patient id:** `{ctx.patient_id}`",
        f"- **Data source:** `{ctx.source}`",
        "",
        "> **Disclaimer:** demonstration only — not a medical device. Do not use for diagnosis or treatment.",
        "",
    ]

    if ctx.source == "csv":
        analysis = pipeline.get("analysis") or {}
        cls = analysis.get("classification") or {}
        screen = analysis.get("screening") or {}
        expl = analysis.get("explanation") or {}
        row_i = pipeline.get("row_index", ctx.row_index)
        lines += [
            "### Emotion model",
            "",
            f"- **Predicted class:** **{cls.get('discrete_emotion', '?')}**",
            f"- **Confidence:** {float(cls.get('confidence', 0)):.1%}",
            "",
            "### Demo risk scores (non-clinical)",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
            f"| Depression (demo) | {float(screen.get('depression_risk', {}).get('score', 0)):.2f}% |",
            f"| Anxiety (demo) | {float(screen.get('anxiety_risk', {}).get('score', 0)):.2f}% |",
            f"| Cognitive load | {float(screen.get('cognitive_load', {}).get('score', 0)):.2f}% |",
            "",
            f"**Recommendation (demo codes):** `{screen.get('recommendation', '')}` — "
            f"{_REC_COPY.get(str(screen.get('recommendation', '')), 'See clinician in real life.')}",
            "",
            "### Explainability (top attributions)",
            "",
        ]
        tops = expl.get("top_features") or []
        for t in tops[:5]:
            nm = t.get("name", "")
            co = float(t.get("contribution", 0))
            lines.append(f"- `{nm}` → contribution **{co:+.4f}**")
        lines.append("")
        nl = expl.get("natural_language_explanation")
        if nl:
            lines += ["### Narrative summary", "", str(nl), ""]
    else:
        screen = pipeline.get("screening") or {}
        if not screen:
            lines += ["_No screening block (train DREAMER VAD if you want VAD-linked screening)._", ""]
        else:
            lines += [
                "### DREAMER epoch (demo)",
                "",
                f"- **Epoch:** {pipeline.get('epoch_index')} · **Subject:** {pipeline.get('subject_id')} · **Trial:** {pipeline.get('trial_id')}",
                "",
                "### Demo risk scores (non-clinical)",
                "",
                "| Metric | Score |",
                "| --- | ---: |",
                f"| Depression (demo) | {float(screen.get('depression_risk', {}).get('score', 0)):.2f}% |",
                f"| Anxiety (demo) | {float(screen.get('anxiety_risk', {}).get('score', 0)):.2f}% |",
            ]
            cog = screen.get("cognitive_load")
            if isinstance(cog, dict) and "score" in cog:
                lines.append(f"| Cognitive load | {float(cog['score']):.2f}% |")
            lines += [
                "",
                f"**Recommendation:** `{screen.get('recommendation', '')}`",
                "",
            ]
        expl = pipeline.get("explanation") or {}
        nl = expl.get("natural_language_explanation")
        if nl:
            lines += ["### Narrative summary", "", str(nl), ""]

    if dashboard_base_url:
        link = build_dashboard_deeplink(dashboard_base_url, ctx)
        lines += [
            "---",
            "",
            "### Interactive dashboard",
            "",
            f"Open the **STRAIN** web UI for this patient (charts, brain view, DREAMER controls):",
            "",
            f"[**Launch STRAIN dashboard →**]({link})",
            "",
            f"If the link is not clickable, copy: `{link}`",
            "",
        ]

    return "\n".join(lines)
