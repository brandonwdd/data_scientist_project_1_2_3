# Project 2: Fraud / Risk Scoring System

Risk Control Scoring and Policy Engine

## Executive Summary

- Predict transaction fraud risk score (risk_score: 0-1)
- Output decisions based on risk score and business rules: APPROVE / REJECT / MANUAL_REVIEW
- Support /score (synchronous) and /score_async (batch)
- Features: data quality gates, MLflow registration/auto-promotion, drift monitoring loop, audit trail
- **Stricter SLO**: p95 ≤ 80ms (vs churn's 120ms)

## Project Structure

```
project_2_fraud_risk_scoring/
├── fraud/
│   ├── configs/
│   │   ├── feature_spec.yaml          # Feature specification
│   │   └── promotion_gate.yaml       # Promotion gate configuration
│   ├── data/
│   │   ├── load_data.py               # Data loading
│   │   └── contracts.py               # Data contracts
│   ├── features/
│   │   ├── engineering.py             # Feature engineering
│   │   └── point_in_time.py           # Point-in-time join
│   ├── models/
│   │   ├── fraud_model.py             # Fraud risk model
│   │   └── explainability.py         # SHAP explainability
│   ├── policy/
│   │   └── policy_engine.py          # Policy engine (decision rules)
│   ├── evaluation/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── protocol.py                # Evaluation protocol
│   ├── serving/
│   │   ├── app.py                     # FastAPI application
│   │   ├── scoring.py                 # Scoring logic
│   │   └── async_tasks.py            # Async task worker
│   ├── training/
│   │   ├── train_fraud.py             # Fraud training script
│   │   └── artifacts.py               # MLflow artifacts generation
│   ├── monitoring/
│   │   ├── drift_job.py              # Drift monitoring (loop)
│   │   └── rollback.py                # Rollback mechanism
│   └── demo/
│       └── demo_5min.py              # 5-minute demo
└── README.md
```

## Core Features

### 1. Risk Scoring Model
- **Baseline**: Isolation Forest (unsupervised anomaly detection)
- **Main**: LightGBM / Random Forest (supervised classification)
- **Calibration**: Isotonic / Platt (probability calibration)

### 2. Policy Engine
- Decision-making based on risk score and business rules
- Decision types:
  - `APPROVE`: Low risk, auto-approve
  - `REJECT`: High risk, auto-reject
  - `MANUAL_REVIEW`: Medium risk, manual review
- Configurable rule priority

### 3. Feature Categories
- **Transaction**: amount, currency, payment_method, merchant_category
- **Behavior**: transaction_count_1h/24h/7d, total_amount_24h, velocity
- **Device & Location**: device_id, ip_country, is_new_device, country_change
- **Risk Signals**: chargeback_count, fraud_flag_count, velocity_risk_score
- **Profile**: account_age, kyc_status, account_status

### 4. Evaluation Metrics
- **Model**: PR-AUC, AUC, TopK Precision@5%/10%, Recall@Precision=90%
- **Policy**: FPR, FNR, Approval Rate, Manual Review Rate

## Quick Start

### 1. Train Model

```bash
# Train Fraud model
python fraud/training/train_fraud.py
```

### 2. Start Service

```bash
# Start FastAPI service
uvicorn fraud.serving.app:app --host 0.0.0.0 --port 8001

# Start Celery worker (async tasks)
celery -A fraud.serving.async_tasks worker --loglevel=info
```

### 3. API Usage

#### Synchronous Scoring
```bash
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_123",
    "amount_usd": 1000.0
  }'
```

#### Asynchronous Batch Scoring
```bash
curl -X POST http://localhost:8001/score_async \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_ids": ["txn_1", "txn_2", "txn_3"],
    "domain": "fraud"
  }'
```

## Differences from Project 1

| Feature | Project 1 (Churn) | Project 2 (Fraud) |
|------|------------------|-------------------|
| Entity | user_id | transaction_id |
| Output | churn_prob, ltv_90d, action | risk_score, decision |
| Decision | NO_ACTION/OFFER_5/OFFER_10 | APPROVE/REJECT/MANUAL_REVIEW |
| SLO p95 | 120ms | **80ms** (stricter) |
| SLO 5xx | 0.5% | **0.1%** (stricter) |
| Cooldown | 14 days | 7 days |
| Core Component | Decisioning Optimizer | Policy Engine |

## Dependencies

- `platform_sdk`: Platform unified SDK
- `fraud.models`: Fraud risk model
- `fraud.policy`: Policy engine

## Configuration

Service uses platform SDK's unified configuration:
- Database connection
- Redis/Celery broker
- MLflow tracking URI
- Logging configuration

## Documentation

For detailed documentation, see:
- `fraud/serving/README.md` - API documentation
- `fraud/configs/promotion_gate.yaml` - Gate configuration
- `fraud/configs/feature_spec.yaml` - Feature specification
