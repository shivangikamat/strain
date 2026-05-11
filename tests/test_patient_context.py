"""Unit tests for demo patient JSON helpers."""

from strain.demo.patient_context import (
    PatientEmotionContext,
    build_dashboard_deeplink,
    example_patient_context_dict,
)


def test_example_patient_context_roundtrip() -> None:
    raw = example_patient_context_dict()
    ctx = PatientEmotionContext.model_validate(raw)
    assert ctx.patient_id
    assert ctx.source == "csv"


def test_dashboard_deeplink_contains_query() -> None:
    ctx = PatientEmotionContext(
        patient_id="p1",
        display_name="Test",
        source="csv",
        row_index=3,
    )
    url = build_dashboard_deeplink("https://example.ngrok-free.app", ctx)
    assert "mode=csv" in url
    assert "row=3" in url
    assert "patientId=p1" in url


def test_markdown_tool_payload_is_json_serializable() -> None:
    """Prompt Opinion expects MCP tool text to be valid JSON for structured tools."""
    import json

    md = "## Title\n\n| Metric | Score |\n| --- | ---: |\n| Demo | 1.0% |\n"
    payload = {"format": "markdown", "markdown": md, "hint": "Render markdown field."}
    s = json.dumps(payload)
    assert "markdown" in s
    assert json.loads(s)["markdown"].startswith("##")
