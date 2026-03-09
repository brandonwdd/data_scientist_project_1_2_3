# Project 1 – Churn & Lifetime Value Decisioning Service

---

### 1. Project Overview – What & Why

- **Goal**: Build a **subscription churn + 90‑day LTV decisioning service** for a SaaS business.
- **Tasks**:
  - Predict whether a user will **churn in the next 30 days**.
  - Estimate the user’s **90‑day Lifetime Value**.
  - Under a **marketing budget constraint**, recommend an action (one of `NO_ACTION`, `OFFER_5`, `OFFER_10`, `CALL_SUPPORT`).
- **Form factor**: An **online API** that product/CRM systems can call to get **scores + recommended action + reasons**.

---

### 2. Data & Problem Setup

- **Data source**: Kaggle SaaS subscription dataset  
  [`https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset/data`](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset/data)
- **Raw data**:
  - Subscription history: plan, start/end dates, billing cycles.
  - Customer attributes and simple engagement signals.
  - Churn status.
- **My problem formulation**:
  - Define a **30‑day churn label** per user at an “as‑of date”.
  - Define a **90‑day LTV target** based on future revenue.
  - Build features so that the same definitions work for **offline training** and **online serving**.

#### 2.1 CSV files

Local data lives under `churn/data/saas_churn_ltv/`. The loader (`churn/data/ravenstack_loader.py`) reads these **5 CSVs**:

| CSV | Rows | Description |
|-----|----------------|--------------|
| `ravenstack_accounts.csv` | 500 | One row per customer (account). |
| `ravenstack_subscriptions.csv` | 5,000 | Subscription periods, plan, MRR/ARR, billing. |
| `ravenstack_churn_events.csv` | 600 | Churn events with date, reason, refund. |
| `ravenstack_feature_usage.csv` | 25,000 | Per-subscription feature usage events. |
| `ravenstack_support_tickets.csv` | 2,000 | Tickets, resolution time, satisfaction. |

**Relations**:

```
                 accounts
                    |
        ---------------------------
        |            |            |
subscriptions   support_tickets   churn_events
        |
   feature_usage
```

#### 2.2 Data structure

| Table | Fields |
|-------|--------|
| **accounts** | `account_id`, `account_name`, `industry`, `country`, `signup_date`, `referral_source`, `plan_tier`, `seats`, `is_trial`, `churn_flag` |
| **subscriptions** | `subscription_id`, `account_id`, `start_date`, `end_date`, `plan_tier`, `seats`, `mrr_amount`, `arr_amount`, `is_trial`, `upgrade_flag`, `downgrade_flag`, `churn_flag`, `billing_frequency`, `auto_renew_flag` |
| **churn_events** | `churn_event_id`, `account_id`, `churn_date`, `reason_code`, `refund_amount_usd`, `preceding_upgrade_flag`, `preceding_downgrade_flag`, `is_reactivation`, `feedback_text` |
| **feature_usage** | `usage_id`, `subscription_id`, `usage_date`, `feature_name`, `usage_count`, `usage_duration_secs`, `error_count`, `is_beta_feature` |
| **support_tickets** | `ticket_id`, `account_id`, `submitted_at`, `closed_at`, `resolution_time_hours`, `priority`, `first_response_time_minutes`, `satisfaction_score`, `escalation_flag` |

---

### 3. System Design & Modules (training vs online serving)

This project lives in `project_1_churn_ltv_decisioning/`. The table below lists **all files in Project 1** and how they are used:

- **train**: offline training / evaluation / backtesting  
- **serve**: online API / async jobs / monitoring  
- **both**: shared by training and serving (e.g., model definitions)

| Category | File / Directory | Usage | Notes |
|---------|------------------|-------|-------|
| Root | `project_1_churn_ltv_decisioning/README.md` | docs | Project overview only |
| Root | `requirements-local.txt` | train | Minimal local dependencies for training |
| Root | `requirements.txt` | both | Full deps incl. platform SDK, MLflow, etc. |
| Root | `.gitignore` | - | Git ignore rules (no runtime impact) |
| Data | `churn/data/__init__.py` | both | Package init (imports only) |
| Data | `churn/data/ravenstack_loader.py` | train | Load 5 CSVs and build churn/LTV labels |
| Data | `churn/data/saas_churn_ltv/ravenstack_accounts.csv` | train | Example account-level table |
| Data | `churn/data/saas_churn_ltv/ravenstack_subscriptions.csv` | train | Example subscriptions + MRR/ARR |
| Data | `churn/data/saas_churn_ltv/ravenstack_churn_events.csv` | train | Example churn events |
| Data | `churn/data/saas_churn_ltv/ravenstack_feature_usage.csv` | train | Example feature-usage events |
| Data | `churn/data/saas_churn_ltv/ravenstack_support_tickets.csv` | train | Example support tickets |
| Data | `churn/data/saas_churn_ltv/README.md` | docs | Data description for local demo |
| Config | `churn/configs/feature_spec.yaml` | train | Feature spec (windows, types, domains) |
| Config | `churn/configs/promotion_gate.yaml` | both | Promotion gate thresholds for model approval |
| Features | `churn/features/__init__.py` | both | Package init |
| Features | `churn/features/ravenstack_features.py` | train | Pandas feature engineering for RavenStack CSVs |
| Features | `churn/features/engineering.py` | train | Generic (Spark-style) feature engineering skeleton |
| Features | `churn/features/point_in_time.py` | both | Point-in-time joins to prevent leakage |
| Models | `churn/models/__init__.py` | both | Package init |
| Models | `churn/models/churn_model.py` | both | 30‑day churn classifier (LogReg / LightGBM + calibration) |
| Models | `churn/models/ltv_model.py` | both | 90‑day LTV regression (LightGBM) |
| Models | `churn/models/explainability.py` | both | SHAP-style explainability / reason codes |
| Models | `churn/models/artifacts/` | train → serve | Directory for saved model artifacts (`*.pkl`) |
| Decisioning | `churn/decisioning/__init__.py` | both | Package init |
| Decisioning | `churn/decisioning/optimizer.py` | both | Combines churn_prob + LTV + costs + budget → `recommended_action` |
| Decisioning | `churn/decisioning/backtest.py` | train | Historical backtest of policies (uplift, profit, offer rate) |
| Evaluation | `churn/evaluation/__init__.py` | train | Package init |
| Evaluation | `churn/evaluation/metrics.py` | train | Model & business metrics (AUC, PR‑AUC, calibration, etc.) |
| Evaluation | `churn/evaluation/protocol.py` | train | Time-based train/val/test split & evaluation protocol |
| Training | `churn/training/__init__.py` | train | Package init |
| Training | `churn/training/train_churn_ravenstack.py` | train | Main local training script (CSV → features → churn+LTV models) |
| Training | `churn/training/train_churn.py` | train | Generic training entrypoint (DataLoader + MLflow) |
| Training | `churn/training/artifacts.py` | train | Organize MLflow artifacts (metrics, plots, model card) |
| Training | `churn/training/promotion_gate.py` | both | Apply promotion gate config to decide if a run can be promoted |
| Serving | `churn/serving/__init__.py` | serve | Package init |
| Serving | `churn/serving/app.py` | serve | FastAPI app exposing `/score`, `/score_async`, `/explain`, `/admin/materialize` |
| Serving | `churn/serving/scoring.py` | serve | `ScoringService`: request → features → models → optimizer → response |
| Serving | `churn/serving/async_tasks.py` | serve | Celery async batch scoring task (`churn.score_batch`) |
| Serving | `churn/serving/README.md` | docs | Serving-specific documentation |
| Monitoring | `churn/monitoring/__init__.py` | serve | Package init |
| Monitoring | `churn/monitoring/drift_job.py` | serve | Data/model drift monitoring job |
| Monitoring | `churn/monitoring/rollback.py` | serve | MLflow-based rollback runbook |
| Demo | `churn/demo/demo_5min.py` | train | 5‑minute end‑to‑end demo (train + serve + call API locally) |

#### 3.8 Services used & Docker containers

- **Services this project uses**:
  - **PostgreSQL**: audit table `churn.prediction_audit`, optional online features; DB name `churn`, schema `churn`.
  - **Redis**: optional (async queue / Celery broker if you use batch jobs).
  - **MLflow**: optional; track runs and register artifacts when training with backend store.
  - **Prometheus**: optional; scrape `/metrics` from the FastAPI app for latency and counts.

- **Docker Compose** (`ds_platform/infra/docker-compose.yml`) defines:

| Service | Image | Port | Purpose |
|---------|--------|----------------|---------|
| postgres | postgres:15-alpine | 5432 | DB for churn/fraud/rag (init.sql creates DBs + schemas). |
| redis | redis:7-alpine | 6379 | Broker / cache. |
| mlflow | ghcr.io/mlflow/mlflow:v2.8.1 | 5000 | Experiment tracking & artifacts. |
| prometheus | prom/prometheus:v2.45.0 | 9090 | Metrics. |
| qdrant | qdrant/qdrant:v1.7.4 | 6333, 6334 | Vector DB (used by Project 3 RAG, not Project 1). |

---

### 4. Key Methods & Design Choices

- **Target design**
  - 30‑day churn as the main retention risk signal.
  - 90‑day LTV to separate “high‑value” vs “low‑value” users under the same churn risk.

- **Feature design**
  - Feature families: customer profile, tenure, usage intensity, billing / revenue statistics.
  - Strong focus on **time semantics** (point‑in‑time features, no peeking into the future).

- **Algorithms used**
  - **Churn model**: Baseline **Logistic Regression** (class_weight=balanced); main **LightGBM** (binary classification, GBDT). **Calibration**: Isotonic or Platt (CalibratedClassifierCV) so outputs are interpretable as probabilities.
  - **LTV model**: **LightGBM regression** (objective=regression, metric=RMSE), optional quantile outputs.
  - **Decisioning**: **Greedy** optimizer (rank by profit_per_cost, consume budget); optional **ILP** (PuLP) for optimal assignment under budget and “at most one offer per user”.
  - **Explainability**: SHAP-style values for reason codes.

- **Decisioning**
  - Instead of only predicting churn, optimize **expected profit**:
    - trade‑off between offer cost and expected incremental LTV.
  - Encode rules in `optimizer.py`, tested via `backtest.py` on historical data.

- **Quality & MLOps**
  - Data contracts (`contracts.py`) and feature spec (`feature_spec.yaml`).
  - Promotion gate (`promotion_gate.yaml`) to prevent bad models from going to “production”.
  - Integration with MLflow and platform SDK for logging and audit.

---

### 5. API & Outputs

- **Synchronous API – `POST /score`**
  - **Input**: `user_id` (+ optional extra context).
  - **Output**:
    - `churn_prob_30d`
    - `ltv_90d`
    - `recommended_action` (one of `NO_ACTION`, `OFFER_5`, `OFFER_10`, `CALL_SUPPORT`)
    - `reason_codes` (e.g. `"high_churn_risk"`, `"high_ltv"`)
    - `request_id` for tracing.

- **Asynchronous API – `POST /score_async`**
  - Batch scoring for many users, returns a job/batch handle.

- **Database / audit outputs**
  - Each `/score` call is written to a Postgres audit table (e.g. `churn.prediction_audit`).
  - Stores input snapshot, model outputs, decision, timestamps, identifiers.

#### 5.1 Postgres schema

- **Logical layout**  
  - **Database**: `churn`  
  - **Schema**: `churn`  
  - **Key tables** (from `ds_platform/infra/postgres/init.sql`):
    - `churn.online_features` – optional online feature store (JSONB), keyed by `(domain, entity_key, feature_set_version)`.
    - `churn.prediction_audit` – one row per `/score` or `/score_async` result; stores predictions and decision as JSONB.
    - `churn.async_jobs` – async job tracking for batch/long-running tasks (status, payload, result).

- **Write path (what happens on `/score`)**  
  1. Client calls `POST /score` with `user_id` and context.  
  2. Service runs feature lookup + churn/LTV models + decision optimizer.  
  3. Service writes a row into `churn.prediction_audit` via the platform SDK, including:
     - `request_id`, `domain="churn"`, `model_name`, `model_version`  
     - `entity_key` (user_id), `predictions` JSON (churn_prob_30d, ltv_90d, etc.)  
     - `decision` JSON (recommended_action, reason_codes) and `latency_ms`, `created_at`.

- **How to query it**
  - Recent 10 decisions:
    ```sql
    SELECT request_id, entity_key, predictions, decision, created_at
    FROM churn.prediction_audit
    WHERE domain = 'churn'
    ORDER BY created_at DESC
    LIMIT 10;
    ```
  - Filter by action, e.g. only offers:
    ```sql
    SELECT request_id, entity_key, decision->>'recommended_action' AS action, created_at
    FROM churn.prediction_audit
    WHERE domain = 'churn'
      AND decision->>'recommended_action' <> 'NO_ACTION';
    ```

- **How it ties back to the API**  
  - Every `/score` response includes a `request_id`; the same `request_id` is the primary key in `churn.prediction_audit`.  
  - Logs and metrics also attach `request_id`, so you can go from **API call → logs → DB row** when debugging or analyzing behavior.

- **Monitoring outputs**
  - Prometheus metrics for latency, error rates, and action distribution.
  - Logs with request IDs to trace issues end‑to‑end.

---

### 6. Results & Business Impact

- **Model performance**
  - Churn model evaluated with ROC‑AUC / PR‑AUC, plus calibration checks.
  - LTV model evaluated with regression metrics (e.g. MAPE / RMSE) and segment‑level sanity checks.

- **Business impact (simulated via backtest)**
  - Compare:
    - baseline “no offers” strategy vs.
    - decisioning strategy from `optimizer.py`.
  - Report:
    - uplift in retained revenue / profit.
    - offer rate and cost per retained user.

---

### 7. Future Directions -- What I Would Do Next

- **Limitations & assumptions**
  - Dataset is synthetic/Kaggle-style; patterns are realistic but do not represent a specific real business.
  - Current modeling focuses on subscription-level tables; richer event logs or product analytics would further improve features.
  - Decision optimizer uses a simplified greedy/ILP formulation; a real system would encode more constraints (channels, caps, fairness, legal, etc.).

- **Model & feature improvements**
  - Try survival / time‑to‑event models or sequence models over subscription history.
  - Enrich behavioral signals if more raw data becomes available.

- **Decisioning & experimentation**
  - Move from fixed thresholds to **contextual bandits** or RL for offer selection.
  - Add an A/B testing layer around the decisioning policy.

- **MLOps & platform integration**
  - Automate data refresh → retrain → evaluation → promotion → deployment.
  - Add stronger drift detection and alerting.
  - Reuse the same patterns in **fraud** and **RAG** projects to build a unified decisioning platform.

- **Potential extensions to production (system design)**
  - Add canary/blue‑green deployments for new model + policy versions, with automatic rollback thresholds.
  - Use an online feature store (e.g. Redis or dedicated FS) so training/serving share the same feature definitions and SLAs.
  - Integrate the `/score` API with CRM/marketing systems (journey orchestration, email/push/SMS) via a thin orchestration layer or event bus.

---

### 8. PowerShell: from clone to run

```powershell
# 1) Optional: start Postgres (for audit table). From repo root:
cd ds_platform\infra
docker compose up -d postgres
cd ..\..

# 2) Enter project and install deps
cd project_1_churn_ltv_decisioning
pip install -r requirements-local.txt
pip install -e ..\ds_platform\platform_sdk

# 3) Train churn (and optionally LTV). Data in churn/data/saas_churn_ltv/
python churn/training/train_churn_ravenstack.py --as-of-date 2024-07-01

# 4) Set env and start API (model paths: adjust if your artifacts live elsewhere)
$env:CHURN_MODEL_PATH="$PWD\churn\\models\\artifacts\churn_model.pkl"
$env:LTV_MODEL_PATH="$PWD\churn\\models\\artifacts\ltv_model.pkl"
$env:POSTGRES_DB="churn"
$env:POSTGRES_SCHEMA="churn"
python -m uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000

# 5) In another terminal: call /score
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/score" -ContentType "application/json" -Body '{"user_id":"A-2e4581"}'
# Or with curl if installed: curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"user_id\": \"A-2e4581\"}"
```

If you use **full infra** (Redis, MLflow, etc.): from `ds_platform/infra` run `docker compose up -d`.

---

# Project 2 – Fraud & Risk Scoring System

---

### 1. Project Overview – What & Why

- **Goal**: Build an **online fraud + risk scoring service** to reduce fraud and chargebacks while controlling customer friction.  
- **Tasks**:
  - Predict a **transaction-level fraud risk score** in \[0, 1\].
  - Convert the risk score into a **business decision**: `APPROVE`, `REJECT`, or `MANUAL_REVIEW`.
  - Meet stricter latency/reliability SLOs than churn: **p95 ≤ 80ms**, **5xx ≤ 0.1%**.
- **Form factor**:
  - FastAPI service with `/score` (sync) and `/score_async` (batch) endpoints.
  - A **policy engine** on top of the model to encode business and risk rules.

---

### 2. Data & Problem Setup

- **Data source**: IEEE-CIS Fraud Detection  
  Reference: [`https://www.kaggle.com/competitions/ieee-fraud-detection/data`](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Raw data**:
  - Transaction and identity CSVs under `project_2_fraud_risk_scoring/fraud/data/ieee_fraud/`.
- **Problem formulation**:
  - **Entity**: `transaction_id`.
  - **Label**: `isFraud` (0 = legitimate, 1 = fraud).
  - **Objective**: produce a **risk score** and a **decision** that balance fraud loss, manual-review cost, and customer experience.

#### 2.1 CSV files

Local data lives under `project_2_fraud_risk_scoring/fraud/data/ieee_fraud/`. The loader (`fraud/data/load_ieee_local.py`) reads these **5 CSVs**:

| CSV | Rows | Description |
|-----|------|-------------|
| `train_transaction.csv` | ~590,000 | Transaction-level features and label `isFraud`. |
| `train_identity.csv` | ~144,000 | Identity/device info; left-joined on `TransactionID`. |
| `test_transaction.csv` | ~506,000 | Test transactions (no label). |
| `test_identity.csv` | ~141,000 | Test identity data. |
| `sample_submission.csv` | 506,691 | Submission format. |

Key ideas: Keep the local demo **simple, numeric, and fast**; cleanly separate local CSV loading from any S3/production data loaders.

---

### 3. System Design & Modules (training vs online serving)

The fraud project is structured similarly to Project 1 but focused on **transactions** and **real-time risk**. The table below lists **all files in Project 2** and how they are used:

- **train**: offline training / evaluation / policy experimentation  
- **serve**: online API / async jobs / monitoring  
- **both**: shared by training and serving (e.g., model definitions, gates)

| Category | File / Directory | Usage | Notes |
|---------|------------------|-------|-------|
| Root | `project_2_fraud_risk_scoring/requirements.txt` | both | Full deps for training + serving (incl. platform SDK) |
| Root | `project_2_fraud_risk_scoring/.gitignore` | - | Git ignore rules (no runtime impact) |
| Data | `fraud/data/__init__.py` | both | Package init |
| Data | `fraud/data/load_data.py` | train | Main data loader (e.g. production/S3 path) |
| Data | `fraud/data/load_ieee_local.py` | train | Local CSV loader for IEEE‑CIS fraud dataset |
| Data | `fraud/data/contracts.py` | train | Data contracts / schema checks |
| Data | `fraud/data/ieee_fraud/train_transaction.csv` | train | Example train transactions (IEEE‑CIS) |
| Data | `fraud/data/ieee_fraud/train_identity.csv` | train | Example train identity/device table |
| Data | `fraud/data/ieee_fraud/test_transaction.csv` | train | Example test transactions (no labels) |
| Data | `fraud/data/ieee_fraud/test_identity.csv` | train | Example test identity data |
| Data | `fraud/data/ieee_fraud/sample_submission.csv` | train | Sample submission format (Kaggle style) |
| Config | `fraud/configs/feature_spec.yaml` | train | Feature spec for fraud (transaction/behavior/device/risk/profile) |
| Config | `fraud/configs/train_local.yaml` | train | Local training config (paths, params) |
| Config | `fraud/configs/promotion_gate.yaml` | both | Promotion gate thresholds for production |
| Config | `fraud/configs/promotion_gate_local.yaml` | train | Local version of promotion gate (demo/experiments) |
| Features | `fraud/features/__init__.py` | both | Package init |
| Features | `fraud/features/engineering.py` | train | Feature engineering for transaction, behavior, device/location, risk, profile |
| Features | `fraud/features/point_in_time.py` | both | Time‑aware joins to avoid leakage across time |
| Models | `fraud/models/__init__.py` | both | Package init |
| Models | `fraud/models/fraud_model.py` | both | Wraps Isolation Forest, Random Forest, LightGBM backends |
| Models | `fraud/models/explainability.py` | both | SHAP‑style explainability utilities |
| Policy | `fraud/policy/__init__.py` | both | Package init |
| Policy | `fraud/policy/policy_engine.py` | both | Rule‑based engine: `risk_score` + key features → `APPROVE/REJECT/MANUAL_REVIEW` |
| Evaluation | `fraud/evaluation/__init__.py` | train | Package init |
| Evaluation | `fraud/evaluation/metrics.py` | train | Model + policy metrics (ROC‑AUC, PR‑AUC, FPR/FNR, etc.) |
| Evaluation | `fraud/evaluation/protocol.py` | train | Evaluation protocol for fair model/policy comparison |
| Training | `fraud/training/__init__.py` | train | Package init |
| Training | `fraud/training/train_fraud.py` | train | Main end‑to‑end training script |
| Training | `fraud/training/artifacts.py` | train | Artifact organization & MLflow logging |
| Serving | `fraud/serving/__init__.py` | serve | Package init |
| Serving | `fraud/serving/app.py` | serve | FastAPI app: `/score`, `/score_async`, `/jobs/{job_id}`, `/health`, `/metrics` |
| Serving | `fraud/serving/scoring.py` | serve | Request → features → model → policy → response |
| Serving | `fraud/serving/async_tasks.py` | serve | Celery‑based async workers for batch scoring |
| Serving | `fraud/serving/README.md` | docs | Serving‑specific documentation |
| Monitoring | `fraud/monitoring/__init__.py` | serve | Package init |
| Monitoring | `fraud/monitoring/drift_job.py` | serve | Drift monitoring loop for fraud models |
| Monitoring | `fraud/monitoring/rollback.py` | serve | Rollback logic when things go wrong |
| Demo | `fraud/demo/demo_5min.py` | train | 5‑minute end‑to‑end fraud demo (train + serve + call API) |

### 3.9 Services used & Docker containers

- **Services this project uses**:
  - **PostgreSQL**: audit (`fraud.prediction_audit`), async job state, optional online features.
  - **Redis**: required for Celery async tasks (broker + result backend).
  - **MLflow** (optional): experiment tracking and artifact storage.
  - **Prometheus** (optional): metrics scraping from `/metrics`.

- **Docker Compose** (`ds_platform/infra/docker-compose.yml`) provides (shared across projects):

| Service   | Image                             | Port            | Purpose                                             |
|-----------|------------------------------------|----------------|-----------------------------------------------------|
| postgres  | postgres:15-alpine                | 5432           | DB for churn/fraud/rag (init.sql creates schemas). |
| redis     | redis:7-alpine                    | 6379           | Celery broker / cache.                             |
| mlflow    | ghcr.io/mlflow/mlflow:v2.8.1      | 5000           | Experiment tracking & artifacts.                   |
| prometheus| prom/prometheus:v2.45.0           | 9090           | Metrics.                                           |
| qdrant    | qdrant/qdrant:v1.7.4              | 6333, 6334     | Vector DB (used by Project 3 RAG).                 |

For the **fraud demo**, you typically need **Postgres + Redis**; MLflow/Prometheus are optional but recommended.

---

### 4. Key Methods & Design Choices

- **Target & task design**
  - Binary classification on `isFraud` at the **transaction** level.
  - Highly imbalanced classes.
  - Objective: good ranking of fraud vs legit **and** good policy metrics.

- **Feature design**
  - **Transaction**: amount, currency, payment method, merchant category.
  - **Behavior**: transaction_count_1h/24h/7d, total_amount_24h, velocity-like features.
  - **Device & Location**: device_id, ip_country, is_new_device, country_change.
  - **Risk Signals**: past chargebacks, past fraud flags, velocity_risk_score.
  - **Profile**: account_age, kyc_status, account_status.

- **Algorithms used**
  - **Baseline**: `IsolationForest` for unsupervised anomaly detection.
  - **Main supervised models**:
    - LightGBM (binary GBDT with imbalance handling).
    - RandomForestClassifier as a simpler alternative.
  - **Calibration**:
    - `CalibratedClassifierCV` with Isotonic or Platt to get calibrated probabilities.
  - **Policy engine**:
    - Configured rules on top of `risk_score` and features (amount, device, country change) to decide APPROVE/REJECT/MANUAL_REVIEW.

---

### 5. API & Outputs

Detailed API docs are in `project_2_fraud_risk_scoring/fraud/serving/README.md`.

- **Synchronous API – `POST /score`**
  - **Input**: `transaction_id` + key features or lookup keys.
  - **Output**:
    - `transaction_id`
    - `risk_score` (0–1)
    - `decision` (`APPROVE`, `REJECT`, `MANUAL_REVIEW`)
    - `reason` / `rule_applied`
    - `request_id`, `model_version`, `feature_set_version`, `latency_ms`

- **Asynchronous API – `POST /score_async`**
  - Takes a list of `transaction_ids` and `domain="fraud"`.
  - Returns a `job_id` for later querying.

- **Job status – `GET /jobs/{job_id}`**
  - Returns job metadata and per-transaction results: risk_score, decision, errors.

- **Database / audit outputs**
  - Each `/score` and completed async job is persisted to **Postgres** in **fraud** DB:
    - `fraud.prediction_audit`: per-request audit rows.
    - `fraud.async_jobs`: async job state and results.

- **Monitoring outputs**
  - `/metrics`: Prometheus metrics (latency, QPS, decision distribution, error rates).
  - Logs: include `request_id` so you can trace from API call → logs → DB.

#### 5.1 Postgres schema & interaction

- **Database**: `fraud`, **schema**: `fraud`.  
- **Key tables**:
  - `fraud.online_features`: optional online features (JSONB).
  - `fraud.prediction_audit`: audit trail of risk scores and decisions.
  - `fraud.async_jobs`: job queue and results for async scoring.

Example queries:

```sql
-- Recent 10 decisions
SELECT request_id, entity_key, predictions, decision, created_at
FROM fraud.prediction_audit
WHERE domain = 'fraud'
ORDER BY created_at DESC
LIMIT 10;

-- All REJECT decisions in last 24h
SELECT request_id, entity_key, decision->>'decision' AS decision, created_at
FROM fraud.prediction_audit
WHERE domain = 'fraud'
  AND decision->>'decision' = 'REJECT'
  AND created_at >= now() - interval '1 day';
```

---

### 6. Results & Business Impact

- **Model performance metrics**:
  - ROC-AUC, PR-AUC on a hold-out set.
  - TopK Precision@5%/10% (precision among top-risk transactions).
  - Recall at fixed precision (Recall@Precision=90%).

- **Policy metrics / business KPIs**:
  - False positive rate (FPR) – legit transactions incorrectly blocked.
  - False negative rate (FNR) – fraud that slips through.
  - Approval Rate, Manual Review Rate.

---

### 7. Future Directions — What I Would Do Next

- **Limitations & assumptions**
  - IEEE-CIS is only a proxy; real production schemas and attack patterns evolve quickly.
  - Features are mostly tabular; richer behavior logs and device intelligence can significantly help.
  - Policy engine is currently rule-based; real setups often involve multiple owners and regulatory constraints.

- **Model & feature improvements**
  - Add sequence/temporal models over transaction histories.
  - Introduce graph-based features (shared cards, devices, IPs, merchants).
  - Explore cost-sensitive learning to reflect true economic cost of errors.

- **Decisioning & experimentation**
  - Add A/B testing and controlled rollouts for new models/policies.
  - Use bandit-style algorithms at decision boundaries (e.g. borderline MANUAL_REVIEW vs APPROVE).

- **MLOps & platform integration**
  - Automate data refresh → retrain → evaluation → promotion → deployment.
  - Strengthen drift detection and alerting on fraud rate, feature distributions, and policy metrics.
  - Integrate with case management / ticketing systems for manual review.

---

### 8. PowerShell: from clone to run

```powershell
# 1) Start Postgres + Redis (for audit + async)
cd ds_platform\infra
docker compose up -d postgres redis
cd ..\..

# 2) Enter project and install deps
cd project_2_fraud_risk_scoring
pip install -r requirements.txt
pip install -e ..\ds_platform\platform_sdk

# 3) Prepare IEEE-CIS data under fraud/data/ieee_fraud/
#    - Required: train_transaction.csv (and optionally train_identity.csv)

# 4) Train fraud model
python fraud/training/train_fraud.py

# 5) Start API service
python -m uvicorn fraud.serving.app:app --host 0.0.0.0 --port 8001

# 6) (Optional) Start Celery worker for async scoring
celery -A fraud.serving.async_tasks worker --loglevel=info

# 7) Call /score from another terminal
Invoke-RestMethod -Method Post -Uri "http://localhost:8001/score" `
  -ContentType "application/json" `
  -Body '{"transaction_id":"txn_123","amount_usd":1000.0}'
```

If you use **full infra** (Redis, MLflow, etc.): from `ds_platform/infra` run `docker compose up -d`.

---

# Project 3 – Enterprise RAG System

---

### 1. Project Overview – What & Why

- **Goal**: Build an **enterprise-grade RAG (Retrieval-Augmented Generation) system** on top of internal documents.  
- **Tasks**:
  - Ingest and index unstructured documents (PDFs, policies, playbooks).  
  - For a user query, **retrieve → rerank → generate** an answer with **grounded citations**.  
  - Use an **evaluation gate** (ragas + custom metrics) to decide whether a configuration is good enough to deploy.
- **Form factor**:
  - FastAPI service under `project_3_enterprise_rag_llm/rag/serving/app.py` with endpoints: `/ask`, `/retrieve`, `/evaluate/run`, `/feedback`.  
  - Uses **Qdrant** as vector store and **Postgres + Redis + Celery** for evaluation and audit.

---

### 2. Data & Problem Setup

- **Data source** (generic, for portfolio description):
  - Internal PDFs and documents (policies, product docs, tickets, FAQs, etc.), stored under `project_3_enterprise_rag_llm/rag/data/pdfs/` for the demo.
- **Derived artifacts**:
  - `chunks.jsonl` – paragraph/section-level chunks with metadata (source_id, page, section, created_at, hash).  
  - Vector index – persisted under `project_3_enterprise_rag_llm/rag/data/index/` and in **Qdrant**.
  - Evaluation set – `eval.jsonl`: list of {query, gold_answer, gold_evidence} for offline evaluation.
- **Problem formulation**:
  - Input: user **query**.  
  - Output: **answer + citations** (which chunks/pages were used) + telemetry (latency, tokens, versions).  
  - Objective: maximize **faithfulness & relevancy**, minimize hallucinations, and optimize **latency & cost**.

#### 2.1 Ingestion & chunking data

- `rag/ingestion/parsers.py`:
  - Parses PDFs via `pymupdf` / `unstructured` into raw text segments.  
- `rag/ingestion/chunker.py`:
  - Splits documents into chunks using **heading hierarchy + sliding windows + overlap**.  
  - Attaches metadata: `source_id`, `page`, `section`, `created_at`, `hash`.  
- Output is written to `chunks.jsonl` or parquet, and later indexed into Qdrant.

Key ideas:

- Preserve enough structure (headings, sections) to support good retrieval and citation.  
- Make chunking configurable so it can be tuned using feedback and evaluation (e.g., via `hard_set.jsonl`).

---

### 3. System Design & Modules (training vs online serving)

The RAG project lives in `project_3_enterprise_rag_llm/`. The table below lists **all key files** and how they are used:

- **train**: offline ingestion / indexing / evaluation / experimentation  
- **serve**: online RAG API, async evaluation jobs, metrics  
- **both**: shared by training and serving (e.g., retrieval/generation pipelines, configs)

| Category | File / Directory | Usage | Notes |
|---------|------------------|-------|-------|
| Root | `project_3_enterprise_rag_llm/README.md` | docs | RAG project overview |
| Root | `project_3_enterprise_rag_llm/requirements.txt` | both | Full deps (LLM client, Qdrant, Celery, MLflow, etc.) |
| Root | `project_3_enterprise_rag_llm/.gitignore` | - | Git ignore rules |
| Root | `project_3_enterprise_rag_llm/env.example` | BOTH | Example `.env` for RAG_LLM_*, QDRANT_*, POSTGRES_*, REDIS_*, CELERY_* |
| RAG core | `rag/__init__.py` | both | Package init |
| Config | `rag/configs/retrieval.yaml` | both | Configures BM25 / dense / hybrid retrieval and reranking, plus index type (e.g. Qdrant) |
| Config | `rag/configs/promotion_gate.yaml` | both | Thresholds for evaluation/promotion gate (faithfulness, relevancy, latency, hallucination rate, etc.) |
| Ingestion | `rag/ingestion/__init__.py` | train | Package init |
| Ingestion | `rag/ingestion/parsers.py` | train | Parses PDFs into raw text segments |
| Ingestion | `rag/ingestion/chunker.py` | train | Splits docs into chunks (headings + sliding windows + overlap) with metadata |
| Retrieval | `rag/retrieval/__init__.py` | both | Package init |
| Retrieval | `rag/retrieval/retriever.py` | both | BM25 / dense / hybrid retrieval + rerank; tracks `index_version`, `retriever_version` |
| Retrieval | `rag/retrieval/index_store.py` | both | Index abstraction for building/storing/loading vector indices |
| Retrieval | `rag/retrieval/qdrant_store.py` | both | Qdrant-backed index implementation |
| Generation | `rag/generation/__init__.py` | both | Package init |
| Generation | `rag/generation/pipeline.py` | both | End‑to‑end generation pipeline (retrieve → prompt → LLM → answer + citations) |
| Generation | `rag/generation/router.py` | both | Routes between cheap FAQ mode and full RAG mode based on query difficulty |
| Generation | `rag/generation/llm_client.py` | both | LLM client abstraction (OpenAI‑compatible; reads env config) |
| Optional | `rag/optional_lora/__init__.py` | train | Placeholder for optional LoRA fine‑tuning integrations |
| Evaluation | `rag/evaluation/__init__.py` | train | Package init |
| Evaluation | `rag/evaluation/metrics.py` | train | RAG metrics (ragas, faithfulness, relevancy, context recall/precision, latency, cost, citation metrics) |
| Evaluation | `rag/evaluation/eval_gate.py` | both | Implements promotion gate using metrics + `promotion_gate.yaml` |
| Evaluation | `rag/evaluation/io.py` | train | I/O utilities for evaluation sets and reports |
| Feedback | `rag/feedback/__init__.py` | train | Package init |
| Feedback | `rag/feedback/hard_set.py` | train | Builds `hard_set.jsonl` from low‑faithfulness / hard cases |
| Serving | `rag/serving/__init__.py` | serve | Package init |
| Serving | `rag/serving/app.py` | serve | FastAPI app exposing `/ask`, `/retrieve`, `/evaluate/run`, `/feedback` |
| Serving | `rag/serving/async_tasks.py` | serve | Celery async tasks for evaluation & background jobs |
| Serving | `rag/serving/rag_metrics.py` | serve | RAG‑specific metrics for Prometheus/Grafana |
| CLI | `rag/cli/__init__.py` | train | Package init |
| CLI | `rag/cli/ingest.py` | train | CLI entrypoint for ingestion & optional index build |
| Artifacts | `rag/artifacts.py` | both | Defines what is logged to MLflow (metrics, ragas reports, latency/cost, index manifest, prompt versions, model card, failure cases) |
| Demo | `rag/demo/demo_5min.py` | train | 5‑minute RAG demo (ingest + build index + start API + call `/ask`) |

---

### 3.9 Services used & Docker containers

- **Services this project uses**:
  - **PostgreSQL**: `rag.prediction_audit`, feedback tables, async jobs metadata.  
  - **Redis + Celery**: async evaluation jobs and background tasks.  
  - **MLflow**: stores RAG evaluation artifacts and configuration snapshots.  
  - **Qdrant**: vector store for dense/hybrid retrieval.  
  - **Prometheus/Grafana**: RAG-specific metrics dashboards.

- **Docker Compose** (`ds_platform/infra/docker-compose.yml`) already includes:
  - Postgres, Redis, MLflow, Prometheus, Qdrant (shared with Projects 1 & 2).

For a full RAG demo, you typically need **all** of: Postgres + Redis + MLflow + Qdrant (Prometheus optional but recommended).

---

### 4. Key Methods & Design Choices

- **Retrieval strategy**
  - Support for **BM25**, **dense embeddings**, and **hybrid + rerank**.  
  - Versioning for retrieval models and prompts (index_version, retriever_version, prompt_version).

- **Generation & guardrails**
  - Prompts enforce **citation format** and `"I don't know"` behavior when evidence is weak.  
  - Router splits between cheap FAQ-mode and full RAG-mode based on query characteristics.  
  - Supports different LLM backends via `RAG_LLM_BASE_URL` / `RAG_LLM_MODEL` (OpenAI-compatible APIs).

- **Evaluation & promotion gate**
  - Uses **ragas** and custom metrics (Evidence Recall@k, Citation Accuracy) to evaluate configurations.  
  - Promotion gate (`promotion_gate.yaml`) encodes hard thresholds on faithfulness, answer relevancy, context recall/precision, citation coverage, latency, and hallucination rate.  
  - Evaluation runs can be asynchronous, with results tracked in MLflow and Postgres.

- **Cost & latency awareness**
  - Evaluation includes latency and token cost metrics; router and retrieval settings can be tuned to meet budgets.  
  - Metrics like `rag_ask_latency_seconds`, `rag_tokens_total` are exported for monitoring.

---

### 5. API & Outputs

- **`POST /ask`**
  - **Input**: `{"query": "...", "options": {...}}` (options may include index version, retrieval mode, etc.).  
  - **Output** (conceptually):
    - `answer`: generated text.  
    - `citations`: list of {source_id, page, chunk_id}.  
    - `meta`: index_version, prompt_version, retriever_version, latency, token usage.  

- **`POST /retrieve`**
  - Returns top-k chunks and their metadata, useful for debugging retrieval quality.

- **`POST /evaluate/run`**
  - Launches an **async evaluation** over an eval set (`eval_set_path`).  
  - Returns `job_id`; progress and results are stored via Celery + Postgres + MLflow.

- **`POST /feedback`**
  - Writes user feedback (e.g. bad answers, missing citations) into a `rag.feedback` table for later analysis / hard set creation.

- **Database / audit outputs**
  - Key tables (per `ds_platform/infra/postgres/init.sql`):
    - `rag.prediction_audit`: audit of `/ask` requests and responses (answer, citations, configs, latency).  
    - `rag.async_jobs`: async evaluation job metadata and results.

---

### 6. Results & Business Impact

- **Quality metrics**:
  - Faithfulness - is answer supported by retrieved evidence?.  
  - Answer relevancy.  
  - Context recall & precision.  
  - Citation coverage & citation accuracy.  

- **Operational metrics**:
  - Latency distribution (p50/p90/p95).  
  - Token cost per query or per session.  
  - Volume of `"I don't know"` responses vs hallucinated answers.

---

### 7. Future Directions — What I Would Do Next

- **Limitations & assumptions**
  - Demo uses a limited set of PDFs; in production you’d integrate S3, Confluence, wikis, ticketing systems, etc.  
  - Current router/reranker can be further tuned or fine-tuned; some behaviors are still rule-based.  
  - Evaluation relies on a curated eval set; coverage and quality of this set are critical.

- **Model & retrieval improvements**
  - Better domain-specific embedding models and rerankers.  
  - Smarter chunking strategies (semantic segmentation, table/image handling).  
  - Multi-index strategies (FAQ index, policy index, long-form index).

- **Decisioning & experimentation**
  - A/B testing different retrieval/rerank/prompt configurations on live traffic.  
  - Bandit-style routing between multiple LLMs or retrieval stacks based on cost/latency/quality signals.

- **MLOps & platform integration**
  - Full automation: ingestion → index build → eval → gate → deploy.  
  - Deeper integration with analytics (dashboards for failure modes, top queries, coverage gaps).  
  - Shared monitoring with Projects 1 & 2 for a unified ML observability story.

---

### 8. PowerShell: from clone to run

```powershell
# 1) Start full infra (includes Postgres, Redis, MLflow, Prometheus, Qdrant)
cd ds_platform\infra
docker compose up -d
cd ..\..

# 2) Enter project and install deps
cd project_3_enterprise_rag_llm
pip install -r requirements.txt
pip install -e ..\ds_platform\platform_sdk

# 3) Configure environment (LLM endpoint, Qdrant, DB, Redis)
Copy-Item env.example .env
# Then edit .env to set RAG_LLM_BASE_URL, RAG_LLM_API_KEY, RAG_LLM_MODEL, QDRANT_HOST/PORT, POSTGRES_*, REDIS_*, CELERY_*

# 4) Ingest PDFs and (optionally) build index
python -m rag.cli.ingest --pdf-dir data\pdfs --out data\chunks.jsonl --build-index --rag-url http://localhost:8002

# 5) Start RAG API service
python -m uvicorn rag.serving.app:app --host 0.0.0.0 --port 8002

# 6) (Optional) Start Celery worker for async evaluation
celery -A rag.serving.async_tasks:celery_app worker -l info -Q celery

# 7) Call /ask from another terminal
Invoke-RestMethod -Method Post -Uri "http://localhost:8002/ask" `
  -ContentType "application/json" `
  -Body '{"query":"Explain our refund policy with citations."}'
```

