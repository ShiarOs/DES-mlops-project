import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import time
from psycopg_pool import ConnectionPool
from src.database import create_prediction_table, save_prediction
from prometheus_client import Counter, Histogram, make_asgi_app
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_URI = "models:/green-taxi-ride-duration@production"
REQUEST_COUNT = Counter(
    "taxi_api_requests_total",
    "Total number of API requests.",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "taxi_api_request_duration_seconds",
    "API request duration in seconds.",
    ["method", "endpoint"],
)

class TripRequest(BaseModel):
    pickup_location_id: int = Field(ge=1)
    dropoff_location_id: int = Field(ge=1)
    trip_distance: float = Field(ge=0.1, le=100)


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )
    mlflow.set_tracking_uri(tracking_uri)

    # app.state.model = mlflow.sklearn.load_model(MODEL_URI)

    # yield
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)

    pool = ConnectionPool(
        conninfo=os.environ["DATABASE_URL"],
        min_size=1,
        max_size=5,
        open=False,
    )
    pool.open()

    try:
        pool.wait(timeout=10)
        create_prediction_table(pool)
        app.state.db_pool = pool
        yield
    finally:
        pool.close()


app = FastAPI(
    title="Green Taxi Duration API",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started_at = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        elapsed_seconds = time.perf_counter() - started_at

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(elapsed_seconds)

@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    trip: TripRequest,
    request: Request,
) -> PredictionResponse:
    model_input = {
        "PULocationID": str(trip.pickup_location_id),
        "DOLocationID": str(trip.dropoff_location_id),
        "trip_distance": trip.trip_distance,
    }

    prediction = request.app.state.model.predict([model_input])[0]
    save_prediction(
        request.app.state.db_pool,
        pickup_location_id=trip.pickup_location_id,
        dropoff_location_id=trip.dropoff_location_id,
        trip_distance=trip.trip_distance,
        predicted_duration_minutes=float(prediction),
    )

    return PredictionResponse(
        predicted_duration_minutes=float(prediction)
    )
