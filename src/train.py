import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "green_tripdata_2024-01.parquet"

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"

EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data was not found at: {DATA_PATH}"
        )

    return pd.read_parquet(DATA_PATH)


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    frame["duration"] = (
        frame["lpep_dropoff_datetime"] - frame["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    frame = frame.loc[
        frame["duration"].between(1, 60)
        & frame["trip_distance"].between(0.1, 100)
    ].copy()

    frame["PULocationID"] = frame["PULocationID"].astype(str)
    frame["DOLocationID"] = frame["DOLocationID"].astype(str)

    return frame[FEATURES + [TARGET]]


def train_and_log() -> tuple[float, str, str]:
    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    raw_data = load_data()
    processed_data = preprocess(raw_data)

    X = processed_data[FEATURES]
    y = processed_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2,
    )

    pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression(),
    )

    with mlflow.start_run() as active_run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": "Shiar_Osman",
                "dataset": "green-taxi",
                "year": "2024",
                "month": "1",
                "features": ", ".join(FEATURES),
                "target": TARGET,
            }
        )

        mlflow.log_params(
            {
                "test_size": 0.2,
                "random_state": 42,
                "duration_min_minutes": 1,
                "duration_max_minutes": 60,
                "distance_min_miles": 0.1,
                "distance_max_miles": 100,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        pipeline.fit(
            X_train.to_dict(orient="records"),
            y_train,
        )

        predictions = pipeline.predict(
            X_test.to_dict(orient="records")
        )
        rmse = root_mean_squared_error(y_test, predictions)

        mlflow.log_metric("rmse", rmse)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="green-taxi-model",
            serialization_format="skops",
        )

        client = MlflowClient()

        try:
            client.create_registered_model(
                name=MODEL_NAME,
                tags={"framework": "scikit-learn"},
                description="Model to predict NYC green taxi trip duration",
            )
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_info.model_uri,
            run_id=active_run.info.run_id,
        )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            version=model_version.version,
            alias="production",
        )

        return (
            rmse,
            active_run.info.run_id,
            model_version.version,
        )


def main() -> None:
    rmse, run_id, model_version = train_and_log()

    print(f"Training data: {DATA_PATH}")
    print(f"RMSE: {rmse:.4f} minutes")
    print(f"MLflow run ID: {run_id}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Model version: {model_version}")
    print("Model alias: production")


if __name__ == "__main__":
    main()