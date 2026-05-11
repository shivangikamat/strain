"""Smoke tests for the FastAPI app (no data files required)."""

from fastapi.testclient import TestClient

from api.main import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
