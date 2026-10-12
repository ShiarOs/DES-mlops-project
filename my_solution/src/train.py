""" 
Contains functions to train a machine learning model to predict taxi ride durations.
"""


from pathlib import Path
import mlflow
import joblib
from mlflow.sklearn import log_model
import pandas as pd
from mlflow import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from my_solution.src.config import YEAR, COLOR, MONTH, RAW_DATA_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, TARGET, FEATURES, MODEL_NAME, PRODUCTION_ALIAS, MODEL_DIR


def setup_mlflow(
    mlflow_tracking_uri: str,
    experiment_name: str,
):
    """
    Sets up the mlflow experiment (if not already there) and instatiates the tracking uri
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
        )
        print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    else:
        print(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
    mlflow.set_experiment(experiment_name)
    

def preprocess(
    df: pd.DataFrame,
    target: str,
    features: list[str],
):
    """
    Preprosses the data to make it ready for training.
    """
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    df = df[features + [target]]
    return df


def train_model(
    df: pd.DataFrame,
    year: int,
    month: int,
    color: str,
    target: str,
    features: list[str],
    model_name: str,
    model_dir: Path
):
    """
    Trains a linear regression model to predict taxi ride durations.
    """
    y = df["duration"]
    X = df.drop(columns=["duration"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    with mlflow.start_run():
        tags = {
            "model": "linear regression",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target,
        }
        mlflow.set_tags(tags)
        
        pipeline = make_pipeline(DictVectorizer(), LinearRegression())
        pipeline.fit(X_train.to_dict(orient="records"), y_train)
        y_pred = pipeline.predict(X_test.to_dict(orient="records"))
        rmse = root_mean_squared_error(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        model_info = log_model(
            sk_model=pipeline, 
            name="mlflow-model", 
            serialization_format="skops",
            registered_model_name=model_name,
            tags={**tags, "rmse": rmse}
        )
        
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_dir / "model.joblib")



def set_production_model(
    alias: str,
    registered_model_name: str,
):
    """
    Gives the alias to the registered model version with the lowest rmse.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    rmse_by_version = {
        version.version: client.get_run(version.run_id).data.metrics.get("rmse", float("inf"))
        for version in versions
        if version.run_id is not None
    }
    if not rmse_by_version:
        raise ValueError(f"No versions with a training run found for '{registered_model_name}'.")
    best_version, best_rmse = min(rmse_by_version.items(), key=lambda item: item[1])
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=best_version,
    )


def main(
    year: int = YEAR,
    month: int = MONTH,
    color: str = COLOR,
    raw_data_path: Path = RAW_DATA_PATH,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    target: str = TARGET,
    features: list[str] = FEATURES,
    model_name: str = MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
    model_dir: Path = MODEL_DIR
):
    """
    Runs the training pipeline: download, preprocess, train and set the production model.
    """
    setup_mlflow(
        mlflow_tracking_uri=mlflow_tracking_uri, experiment_name=experiment_name
    )
    data_path = raw_data_path / f"{color}_tripdata_{year}-{month:02d}.parquet"
    df = preprocess(pd.read_parquet(data_path), target=target, features=features)
    train_model(
        df,
        year=year,
        month=month,
        color=color,
        target=target,
        features=features,
        model_name=model_name,
        model_dir=model_dir
    )
    set_production_model(alias=alias, registered_model_name=model_name)
    

if __name__ == "__main__":
    main()