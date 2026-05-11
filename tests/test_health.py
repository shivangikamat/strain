"""Smoke tests for the FastAPI app (no data files required)."""

from fastapi.testclient import TestClient

from api.main import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_screen_mental_health_has_cognitive_load() -> None:
    from strain.screening.mental_health import screen_mental_health

    cls = {"probabilities": {"NEGATIVE": 0.1, "NEUTRAL": 0.2, "POSITIVE": 0.7}, "confidence": 0.7}
    feats = {"spectral_ratios": {"beta_alpha": 2.1, "theta_alpha": 0.9}}
    result = screen_mental_health(cls, feats)
    assert "cognitive_load" in result
    assert 0 <= result["cognitive_load"]["score"] <= 100


def test_dreamer_vad_screening_has_cognitive_load() -> None:
    from strain.models.dreamer_vad import dreamer_vad_screening

    pred = {"valence": 3.0, "arousal": 4.0, "dominance": 2.5}
    result = dreamer_vad_screening(pred, beta_alpha=2.1)
    assert "cognitive_load" in result
    assert 0 <= result["cognitive_load"]["score"] <= 100
