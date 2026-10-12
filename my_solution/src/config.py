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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green_taxi_linear_regression"
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"
PRODUCTION_ALIAS = "production"