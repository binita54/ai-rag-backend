"""Smoke test for the FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    """The /health endpoint should return 200 with a status payload."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
