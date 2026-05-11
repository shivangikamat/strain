"""
Demo-only mental health style scores from emotion probabilities + spectral ratios.

NOT clinical — hackathon / education only.
"""

from __future__ import annotations

from typing import Any


def screen_mental_health(
    classification: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probs = classification.get("probabilities", {})
    p_neg = float(probs.get("NEGATIVE", 0.0))
    p_neu = float(probs.get("NEUTRAL", 0.0))
    p_pos = float(probs.get("POSITIVE", 0.0))

    ratios = (features or {}).get("spectral_ratios", {})
    beta_alpha = float(ratios.get("beta_alpha", 1.0))

    depression_risk = min(100.0, max(0.0, 55.0 * p_neg + 8.0 * max(0.0, beta_alpha - 1.0)))
    anxiety_risk = min(100.0, max(0.0, 45.0 * (1.0 - p_neu) + 10.0 * max(0.0, beta_alpha - 1.0)))
    cognitive_load = min(100.0, max(0.0, beta_alpha / 3.0 * 100.0))

    rec = "no_concern"
    if depression_risk > 70 or anxiety_risk > 70:
        rec = "seek_specialist"
    elif depression_risk > 50 or anxiety_risk > 50:
        rec = "consult_pcp"
    elif depression_risk > 35 or anxiety_risk > 35:
        rec = "monitor"

    return {
        "disclaimer": (
            "Demonstration scores only — not a medical device. "
            "Do not use for diagnosis or treatment decisions."
        ),
        "depression_risk": {"score": depression_risk, "confidence": classification.get("confidence")},
        "anxiety_risk": {"score": anxiety_risk, "confidence": classification.get("confidence")},
        "cognitive_load": {"score": cognitive_load},
        "recommendation": rec,
        "key_findings": [
            f"Emotion model: NEGATIVE {p_neg:.2f}, NEUTRAL {p_neu:.2f}, POSITIVE {p_pos:.2f}.",
            f"Proxy beta/alpha ratio: {beta_alpha:.3f}.",
        ],
    }
