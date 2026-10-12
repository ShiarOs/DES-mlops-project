"""Smoke tests for the prediction API."""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_TRIP = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.5}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_plausible_duration():
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    assert 1 <= response.json()["duration_minutes"] <= 60


def test_predict_rejects_negative_distance():
    response = client.post("/predict", json={**VALID_TRIP, "trip_distance": -1})
    assert response.status_code == 422

def test_metrics_endpoint_exposes_prediction_counter():
    client.post("/predict", json=VALID_TRIP)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "taxi_predictions_total" in response.text

