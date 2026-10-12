"""Prediction API for the green taxi trip-duration model."""

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"


class TripRequest(BaseModel):
    """One taxi trip, described by the same three features the model was trained on."""

    PULocationID: int = Field(..., ge=1, le=265, examples=[74])
    DOLocationID: int = Field(..., ge=1, le=265, examples=[75])
    trip_distance: float = Field(..., gt=0, le=100, allow_inf_nan=False, examples=[1.5])


class TripPrediction(BaseModel):
    """The model's answer."""

    duration_minutes: float


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No model at {MODEL_PATH}. Run `uv run dvc repro` first.")
MODEL = joblib.load(MODEL_PATH)

app = FastAPI(title="Green Taxi Trip Duration API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Answer 'ok' if the API is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: TripRequest) -> TripPrediction:
    """Predict the duration of one trip in minutes."""
    features = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    duration = float(MODEL.predict([features])[0])
    return TripPrediction(duration_minutes=round(duration, 2))

