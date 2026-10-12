"""Load and clean the NYC green taxi trip data."""

from pathlib import Path

import pandas as pd

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def load_data(path: Path) -> pd.DataFrame:
    """Read one month of taxi trips from a parquet file."""
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add trip duration in minutes, drop outliers, keep only model columns."""
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    return df[FEATURES + [TARGET]]

