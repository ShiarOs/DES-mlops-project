# Taxi MLOps Project: Implementation Report

## 1. Project Overview

This project turns the supplied NYC Green Taxi training notebook into a small local-first MLOps workflow. It covers reusable training, experiment and model tracking, local DVC data versioning, prediction serving, API metrics, a local monitoring stack, PostgreSQL prediction logging, and an Evidently drift report.

The model predicts completed trip duration from pickup zone, dropoff zone, and `trip_distance`. The distance is the taximeter distance recorded during the completed trip. It is suitable for historical replay and monitoring exercises, not as a pre-trip duration estimator.

## 2. Training Data and Baseline

The notebook uses January 2024 NYC Green Taxi data. The downloaded Parquet file contained 56,551 rows and 20 columns. Preprocessing calculated duration in minutes and retained trips with duration from 1 through 60 minutes and distance from 0.1 through 100 miles. It retained 52,386 rows.

The selected features are `PULocationID`, `DOLocationID`, and `trip_distance`; the target is `duration`. Location IDs are categorical identifiers and are converted to strings. The model is a scikit-learn pipeline with `DictVectorizer` and `LinearRegression`. The data is split with a 20% test set and random state 42.

Baseline and refactored training produced an RMSE of 5.0688 minutes. A successful refactored run recorded an MLflow run and registered model version 3 under the `production` alias at that point in the project's history. Later successful training executions may have created newer versions.

## 3. Files and Responsibilities

### Training and project configuration

- `01-train-ml-model.ipynb`: supplied baseline notebook. It downloads and explores the data, preprocesses it, trains the baseline, evaluates it, and demonstrates MLflow tracking and model registration.
- `src/__init__.py`: makes `src` importable as a Python package.
- `src/train.py`: loads the taxi dataset, preprocesses it, creates the reproducible split, trains the pipeline, records parameters and RMSE in MLflow, logs the model, registers a model version, and updates the `production` alias.
- `pyproject.toml`: declares project dependencies, including DVC, Evidently, FastAPI, MLflow, scikit-learn, Prometheus client, and PostgreSQL driver/pool support.
- `uv.lock`: pins the resolved dependency graph for repeatable installation with `uv sync --locked`.
- `.python-version`: records the project's Python version selection.
- `README.md` and `project-description.md`: explain the learning objectives, expected deliverables, and project milestones.

### Data versioning

- `data/green_tripdata_2024-01.parquet`: local training data; Git ignores the large data file.
- `data/green_tripdata_2024-01.parquet.dvc`: DVC metadata containing the tracked data checksum and size.
- `data/.gitignore`: prevents Git from tracking the large data object directly.
- `.dvc/config`: project DVC configuration. The local remote path was configured with a local-only setting and is not shared through Git.
- `.dvcignore`: DVC ignore rules.
- `.gitignore`: excludes local datasets, MLflow stores, and other generated/local files while retaining DVC metadata.

The configured DVC remote is a directory on the developer's computer. `dvc push` and `dvc status -c` succeeded locally. This is not a shared remote and does not distribute data to another machine.

### API and database

- `src/api.py`: FastAPI application. Its lifespan loads the model from MLflow and opens a PostgreSQL connection pool. `/health` reports model loading, `/predict` validates a request and returns a duration prediction, and `/metrics` exposes Prometheus metrics. Middleware records request counts and durations.
- `src/database.py`: creates the `prediction_logs` table and inserts each prediction's timestamp, location IDs, distance, and predicted duration.
- `.env.example`: safe template for local configuration; it must contain placeholders only.
- `.env`: local-only configuration for MLflow and PostgreSQL. It is ignored by Git and must not be committed or copied into reports.

The API returned approximately 10.9503 minutes for pickup zone 236, dropoff zone 239, and distance 1.98 miles. The example's observed trip duration in the notebook was 11.5 minutes. This single example is not a model evaluation; the held-out RMSE is the relevant aggregate metric.

### Monitoring, reporting, and automation

- `docker-compose.yaml`: defines local Prometheus, Grafana, and PostgreSQL services with persistent Docker volumes. The API itself is started directly with Uvicorn on the host rather than as a Compose service.
- `monitoring/prometheus.yml`: configures Prometheus to scrape the host API at `host.docker.internal:8000` every five seconds.
- `monitoring/grafana/provisioning/datasources/datasource.yml.off`: retained datasource provisioning definition. It was renamed to `.off` during diagnosis because provisioning caused Grafana startup failures.
- `monitoring/grafana/provisioning/dashboards/provider.yml.off`: retained dashboard provider definition, also disabled during diagnosis.
- `monitoring/grafana/dashboards/taxi-api.json`: dashboard definition with request count, request rate, and p95 prediction latency panels. Automatic dashboard provisioning is currently disabled by the `.off` provider filename.
- `src/monitor.py`: loads a 5,000-row reference sample from processed training data, reads up to 500 recent prediction inputs from PostgreSQL, requires at least 30 current requests, and generates an Evidently data drift report.
- `reports/data-drift.html`: generated interactive HTML drift report.
- `.github/workflows/taxi-training.yaml`: GitHub Actions workflow for Python 3.13. It installs locked dependencies, downloads the public January 2024 dataset, checks it against DVC metadata, runs training, and uploads local MLflow files as a workflow artifact.
- `.github/workflows/reference-solution.yaml`: pre-existing workflow for the separate reference example; it is not the taxi workflow.
- `assets/`: supplied visual and educational assets.

## 4. Local Architecture and Data Flow

1. DVC metadata identifies the expected taxi training dataset. The developer's local DVC cache/remote stores its data object locally.
2. `src/train.py` loads the dataset, derives duration, filters outliers, splits the data, trains the pipeline, evaluates RMSE, and records the run and model in local MLflow tracking storage.
3. `src/api.py` loads the registered model and serves validated prediction requests.
4. Each successful prediction is inserted into PostgreSQL by `src/database.py`.
5. Prometheus scrapes API request counters and latency histograms from `/metrics`.
6. Grafana queries Prometheus for operational monitoring.
7. `src/monitor.py` compares reference training features with recent request features and saves an Evidently report.

## 5. Verification Results

- The notebook and refactored training completed successfully.
- A refactored training run reported RMSE 5.0688 minutes and registered the model.
- DVC reported data and local remote in sync; a local `dvc push` completed.
- API health returned `status: ok` and `model_loaded: true` during successful runs.
- API inference returned approximately 10.9503 minutes for the sample request.
- Prometheus scraped the API successfully. The prediction histogram p95 query returned 0.075 seconds after test traffic was sent.
- PostgreSQL accepted connections and stored prediction inputs and outputs.
- Evidently generated a report using 5,000 reference rows and 31 current prediction rows.
- The report detected drift in all three monitored columns: pickup location, dropoff location, and trip distance. The current dataset was made from synthetic test requests, so this validates the monitoring workflow and must not be presented as evidence of real-world production drift.
- The user reported completion of the GitHub Actions assignment. The workflow file is present locally, but its successful remote run was not independently inspected in this workspace session.

## 6. Problems Encountered and Resolutions

### Pandas categorical assignment

The first preprocessing implementation assigned strings into existing `int32` columns using a multi-column `.loc` assignment. Pandas rejected the lossy dtype conversion. The code was corrected to replace each location column directly with its `astype(str)` result.

### Google Drive dependency

DVC's Google Drive extra was briefly installed. The project requirement changed to avoid paid or cloud-hosted storage. The extra was removed, the lockfile was regenerated, and the Google Drive-specific packages were uninstalled. DVC remains configured with local storage.

### Grafana datasource schema warning and restart loop

The datasource YAML was initially named `prometheus.yml`, causing the editor to apply a Prometheus schema and warn that `apiVersion` was not allowed. The file was renamed to `datasource.yml`.

Grafana later restarted during provisioning with `Datasource provisioning error: data source not found`. The provisioning files were disabled by renaming them to `.off`, after which Grafana could start. The datasource and dashboard were then handled through the Grafana UI. The dashboard JSON exists, but its automatic provider is currently disabled and should be repaired before claiming fully repeatable Grafana provisioning.

### PostgreSQL connection pool timeout

The database accepted connections from within its container, but the Python connection pool timed out when using `localhost` on Windows. A direct test showed that the pool connected successfully with IPv4 loopback, so the database host in `DATABASE_URL` was changed to `127.0.0.1`.

The original database password contained `@`, which must be URL-encoded as `%40` within a connection URI. Credentials were exposed in conversation during debugging; rotate them before sharing or publishing the project, and keep `.env` out of Git.

### Evidently sample threshold

Report generation intentionally requires at least 30 current predictions. The initial run failed with a clear `ValueError` because too few requests had been logged. A batch of varied test requests was sent, after which the report completed with 31 records.

### Prometheus histogram and missing chart line

The API metric endpoint initially displayed metric declarations but no series until requests were made. The p95 query first returned `NaN` with a single prediction observation. After multiple predictions and additional scrape intervals, it returned 0.075 seconds.

## 7. Reproduction Commands

Install dependencies and train:

```powershell
uv sync --locked
uv run python -m src.train
```

Run the API in one terminal:

```powershell
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Start local monitoring and database services:

```powershell
docker compose up -d prometheus grafana postgres
```

Generate an Evidently report after at least 30 predictions have been recorded:

```powershell
uv run python -m src.monitor
```

The report is saved to `reports/data-drift.html`. Never commit `.env`, real credentials, or database passwords.

## 8. Scope and Remaining Caveats

- DVC storage is local-only. A different machine cannot retrieve the DVC object from this setup. GitHub Actions downloads the public dataset directly and validates it against the committed DVC metadata.
- MLflow uses local SQLite/file artifacts by default. The workflow uploads these files as a run artifact; it does not configure a shared MLflow server or PostgreSQL-backed MLflow store.
- PostgreSQL stores prediction inputs and model outputs. It does not currently store actual post-trip durations, so live prediction accuracy cannot be calculated from these rows.
- The Evidently report is generated as a batch command, not continuously from the API request path.
- Grafana datasource/dashboard provisioning definitions are disabled with `.off` suffixes after the startup issue. The Grafana UI may retain manually configured resources in its persistent volume, but a fresh deployment will not provision them from those files until the error is fixed.
- The API is not yet containerized in `docker-compose.yaml`; Prometheus reaches the host-run API through `host.docker.internal`.
- The model uses completed-trip distance, which limits its interpretation to historical replay and monitoring.