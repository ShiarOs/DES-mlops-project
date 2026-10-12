"""
Contains pre-defined variables used in model training
"""


# Imports
from pathlib import Path
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()


# Variables
YEAR = 2024
MONTH = 1
COLOR = "green"
PARENT_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PARENT_PATH / "data" / "raw"
MODEL_DIR = PARENT_PATH / "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PARENT_PATH / 'mlflow.db'}")
MLFLOW_ARTIFACT_LOCATION_URI = (PARENT_PATH / "mlruns").as_uri()
EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green_taxi_linear_regression"
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"
PRODUCTION_ALIAS = "production"