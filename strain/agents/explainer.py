"""Explainer agent — thin wrapper (explanations produced in AnalysisEngine)."""


class Explainer:
    """Reserved for richer NLG; orchestrator uses AnalysisEngine explanation."""

    def summarize(self, analysis: dict) -> str:
        cls = analysis.get("classification", {})
        exp = analysis.get("explanation", {})
        return exp.get(
            "natural_language_explanation",
            f"Predicted: {cls.get('discrete_emotion', 'unknown')}",
        )
