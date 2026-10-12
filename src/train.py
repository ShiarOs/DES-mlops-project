"""Train the taxi trip-duration model and save it to disk."""

from pathlib import Path

import joblib
import click

import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.preprocess import FEATURES, TARGET, load_data, preprocess

from src.prepare_data import data_path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

EXPERIMENT_NAME = "green-taxi-monitoring"

REGISTERED_MODEL_NAME = "green-taxi-ride-duration"


def split_data(df):
    """Split into 80% training and 20% test rows, the same way as the notebook."""
    return train_test_split(df[FEATURES], df[TARGET], random_state=42, test_size=0.2)


def train(X_train, y_train):
    """One-hot encode the zones, then fit a linear regression."""
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(X_train.to_dict(orient="records"), y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test) -> float:
    """Return the RMSE in minutes on the test rows."""
    y_pred = pipeline.predict(X_test.to_dict(orient="records"))
    return root_mean_squared_error(y_test, y_pred)


def setup_mlflow():
    """Point MLflow at the tracking store from .env and select the experiment."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)


@click.command()
@click.option("--year", default=2024, show_default=True, help="Year of the training data.")
@click.option(
    "--month",
    default=1,
    show_default=True,
    type=click.IntRange(1, 12),
    help="Month of the training data (1-12).",
)
def main(year: int, month: int):
    setup_mlflow()
    df = preprocess(load_data(data_path(year, month)))
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run():
        mlflow.set_tags({"model": "linear regression", "dataset": "green-taxi"})
        mlflow.log_params({"year": year, "month": month, "features": FEATURES, "test_size": 0.2})

        pipeline = train(X_train, y_train)
        rmse = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse:.6f}")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="mlflow-model",
            serialization_format="skops",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        MlflowClient().set_registered_model_alias(
            REGISTERED_MODEL_NAME, "production", model_info.registered_model_version
        )
        print(f"Registered {REGISTERED_MODEL_NAME} version {model_info.registered_model_version} as 'production'")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

