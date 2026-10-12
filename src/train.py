"""Train the taxi trip-duration model and save it to disk."""

from pathlib import Path

import joblib
import click

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.preprocess import FEATURES, TARGET, load_data, preprocess

from src.prepare_data import data_path

ROOT_DIR = Path(__file__).resolve().parents[1]

#DATA_PATH = ROOT_DIR / "data" / "green_tripdata_2024-01.parquet"
#MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

#DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

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
    df = preprocess(load_data(data_path(year, month)))
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train(X_train, y_train)
    rmse = evaluate(pipeline, X_test, y_test)
    print(f"RMSE: {rmse:.6f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

