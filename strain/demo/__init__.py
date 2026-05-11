"""Demo helpers: patient context JSON, dashboard deep links, Markdown reports."""

from strain.demo.patient_context import (
    PatientEmotionContext,
    build_dashboard_deeplink,
    example_patient_context_dict,
    format_screening_markdown,
    parse_patient_context_json,
    run_emotion_pipeline_for_context,
)

__all__ = [
    "PatientEmotionContext",
    "build_dashboard_deeplink",
    "example_patient_context_dict",
    "format_screening_markdown",
    "parse_patient_context_json",
    "run_emotion_pipeline_for_context",
]
