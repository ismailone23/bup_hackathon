from fastapi.testclient import TestClient

from main import app


def test_ismail_health_is_ready_when_api_key_exists(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
