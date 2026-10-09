"""
Contains functions to download data and save to disk.
"""

from pathlib import Path
from my_solution.src.config import YEAR, MONTH, COLOR, RAW_DATA_PATH
from urllib.request import urlretrieve


def download_data(
    year: int,
    month: int,
    color: str,
    raw_data_path: Path,
):
    """
    Downloads data from the url, saves it to the specified path and returns the file path.
    """
    data_path = raw_data_path / f"{color}_tripdata_{year}-{month:02d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        urlretrieve(
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet",
            data_path,
        )
        print("Downloaded data.")
    else:
        print("Data already exists in folder, skipping downloading.")


def main(
    year=YEAR,
    month=MONTH,
    color=COLOR,
    raw_data_path=RAW_DATA_PATH
):
    download_data(
        year=year, month=month, color=color, raw_data_path=raw_data_path
    )


if __name__ == "__main__":
    main()