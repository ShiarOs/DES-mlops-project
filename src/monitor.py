import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

from src.train import DATA_PATH, FEATURES, preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "data-drift.html"


def load_reference_data() -> pd.DataFrame:
    raw_data = pd.read_parquet(DATA_PATH)
    processed_data = preprocess(raw_data)
    reference_data = processed_data[FEATURES].copy()

    return reference_data.sample(
        n=min(5000, len(reference_data)),
        random_state=42,
    )


def load_current_data() -> pd.DataFrame:
    query = """
        SELECT
            pickup_location_id AS "PULocationID",
            dropoff_location_id AS "DOLocationID",
            trip_distance
        FROM prediction_logs
        ORDER BY created_at DESC
        LIMIT 500
    """

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        rows = connection.execute(query).fetchall()

    return pd.DataFrame(rows, columns=FEATURES)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    reference_data = load_reference_data()
    current_data = load_current_data()

    if len(current_data) < 30:
        raise ValueError(
            "Log at least 30 prediction requests before generating the report."
        )

    current_data["PULocationID"] = current_data["PULocationID"].astype(str)
    current_data["DOLocationID"] = current_data["DOLocationID"].astype(str)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    print(f"Reference rows: {len(reference_data)}")
    print(f"Current prediction rows: {len(current_data)}")
    print(f"Drift report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()