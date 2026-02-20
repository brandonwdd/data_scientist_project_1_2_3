# Project 1: Churn + LTV Decisioning Service

Subscription Churn and Lifetime Value Decisioning System

## Executive Summary

- Predict 30d churn_prob + 90d LTV
- Output recommended_action + reason_codes under budget constraints
- Support /score (synchronous) and /score_async (batch)
- Features: data quality gates, MLflow registration/auto-promotion, drift monitoring, audit trail

## Project Structure

```
project_1_churn_ltv_decisioning/
├── churn/
│   ├── configs/
│   │   ├── feature_spec.yaml          # Feature specification
│   │   └── promotion_gate.yaml        # Promotion gate configuration
│   ├── data/
│   │   ├── load_data.py               # Data loading
│   │   └── contracts.py                # Data contracts
│   ├── features/
│   │   ├── engineering.py             # Feature engineering
│   │   └── point_in_time.py           # Point-in-time join
│   ├── models/
│   │   ├── churn_model.py             # Churn model
│   │   ├── ltv_model.py               # LTV model
│   │   └── explainability.py          # SHAP explainability
│   ├── decisioning/
│   │   ├── optimizer.py               # Budget constraint optimizer
│   │   └── backtest.py                # Backtest framework
│   ├── evaluation/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── protocol.py                # Evaluation protocol
│   ├── serving/
│   │   ├── app.py                     # FastAPI application
│   │   └── scoring.py                 # Scoring logic
│   └── training/
│       ├── train_churn.py             # Churn training script
│       ├── train_ltv.py               # LTV training script
│       └── artifacts.py               # MLflow artifacts generation
└── README.md

```

## Quick Start

### Local run (RavenStack data, no S3/Spark)

1. **Enter project_1_churn_ltv_decisioning and install dependencies** (project deps + platform_sdk; both required):

   **Python 3.13**: Use minimal requirements first to avoid building pyarrow/pyspark; add full deps after local run works.

```bash
cd project_1_churn_ltv_decisioning
pip install -r requirements-local.txt
pip install -e ../ds_platform/platform_sdk
```

   For full environment (MLflow, Spark, etc.), then run: `pip install -r requirements.txt`

2. **Train Churn + LTV** (data in `churn/data/saas_churn_ltv/`):

```bash
python churn/training/train_churn_ravenstack.py --as-of-date 2024-07-01
```

3. **Start the service and set model paths and DB**:

```bash
# Windows CMD
set CHURN_MODEL_PATH=%cd%\data\demo_models\churn_model.pkl
set LTV_MODEL_PATH=%cd%\data\demo_models\ltv_model.pkl
set POSTGRES_DB=churn
set POSTGRES_SCHEMA=churn
uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000
```

```powershell
# PowerShell
$env:CHURN_MODEL_PATH="$PWD\data\demo_models\churn_model.pkl"
$env:LTV_MODEL_PATH="$PWD\data\demo_models\ltv_model.pkl"
$env:POSTGRES_DB="churn"
$env:POSTGRES_SCHEMA="churn"
python -m uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000
```

4. **Call /score** (in another terminal):

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"user_id\": \"A-2e4581\"}"
```

---

### Local PostgreSQL (audit persistence)

On each `/score` call, the service writes the request to **local PostgreSQL** in the **churn** database, **churn** schema, table `churn.prediction_audit`, via platform_sdk, for auditing and debugging.

**The three DBs under server postgres_ds are fully separate**: churn DB has only churn tables, fraud only fraud tables, rag only rag tables. Postgres is orchestrated in ds_platform; use pgAdmin on the host to connect to the Postgres in Docker.

1. **Start local Postgres** (under ds_platform):

```bash
cd ds_platform/infra
docker compose up -d postgres
```

   For Redis, MLflow, etc. run: `docker compose up -d`

2. **Environment variables** (optional; defaults used if unset):

   - `POSTGRES_HOST`: default `localhost`
   - `POSTGRES_PORT`: default `5432`
   - `POSTGRES_USER`: default `mlplatform`
   - `POSTGRES_PASSWORD`: default `mlplatform_dev`
   - **`POSTGRES_DB`**: for project_1_churn_ltv_decisioning set to **`churn`**
   - **`POSTGRES_SCHEMA`**: for project_1_churn_ltv_decisioning set to **`churn`** (same as DB name; tables like churn.prediction_audit)

   Table schema is created automatically on first Postgres start via `init.sql` (all three DBs churn / fraud / rag are created).

3. **Starting the service** requires no code changes; if Postgres is down or unreachable, audit writes fail with logs but do not affect the `/score` HTTP response.

4. **Inspect audit data** (after connecting to churn DB in pgAdmin):

```sql
SELECT request_id, domain, entity_key, predictions, decision, created_at
FROM churn.prediction_audit
ORDER BY created_at DESC
LIMIT 10;
```

---

### Default path (S3 + Spark, not used currently)

### 1. Train Models

```bash
# Train Churn model
python churn/training/train_churn.py

# Train LTV model
python churn/training/train_ltv.py
```

### 2. Start Service

```bash
uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000
```

### 3. Test API

```bash
# Synchronous scoring
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"user_id": "12345"}'

# Asynchronous batch scoring
curl -X POST http://localhost:8000/score_async \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["12345", "67890"], "domain": "churn"}'
```

## 5-Min Demo

1. Start infrastructure: `cd infra && docker compose up -d`
2. Train model: `python churn/training/train_churn.py`
3. Check MLflow: http://localhost:5000 (view runs and artifacts)
4. Start service: `uvicorn churn.serving.app:app`
5. Test scoring: call `/score` API
6. View metrics: http://localhost:9090 (Prometheus)

## Document Sections

- **B1-B2**: Scope & Non-goals
- **B3**: Time Semantics & Label Definition
- **B4**: Data & Contracts
- **B5**: Feature Store (Offline + Online)
- **B6**: Modeling
- **B7**: Decisioning Layer
- **B8**: Offline Evaluation Protocol
- **B9**: Serving & APIs
- **B10**: Promotion Gate
- **B11**: Drift & Retrain & Rollback
- **B12**: MLflow Artifacts
- **B13**: 5-Min Demo
