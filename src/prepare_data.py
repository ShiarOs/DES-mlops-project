"""Download one month of NYC green taxi trip data."""

from pathlib import Path
from urllib.request import urlretrieve

import click

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def data_path(year: int, month: int) -> Path:
    """Location of one month of green taxi data, e.g. data/green_tripdata_2024-01.parquet."""
    return DATA_DIR / f"green_tripdata_{year}-{month:02d}.parquet"


def download_data(year: int, month: int) -> Path:
    """Download one month of data, unless the file is already on disk."""
    path = data_path(year, month)
    if path.exists():
        print(f"Already downloaded: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"{BASE_URL}/{path.name}", path)
    print(f"Downloaded: {path}")
    return path


@click.command()
@click.option("--year", default=2024, show_default=True, help="Year of the data.")
@click.option(
    "--month",
    default=1,
    show_default=True,
    type=click.IntRange(1, 12),
    help="Month of the data (1-12).",
)
def main(year: int, month: int):
    download_data(year, month)


if __name__ == "__main__":
    main()

