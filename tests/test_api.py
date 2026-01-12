from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_ok():
    r = client.post("/api/v1/predict", json={"text": "это было прекрасно"})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data
    assert "probabilities" in data


def test_predict_validation():
    r = client.post("/api/v1/predict", json={"text": ""})
    assert r.status_code == 422
