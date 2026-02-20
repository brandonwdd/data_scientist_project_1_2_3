# Fraud Serving Service

## Overview

Fraud / Risk Scoring Service provides synchronous and asynchronous scoring endpoints using the platform SDK's unified capabilities.

## API Endpoints

### 1. POST `/score` - Synchronous Scoring
Score a single transaction synchronously, returns risk_score, decision (APPROVE/REJECT/MANUAL_REVIEW), and reason.

**Request**:
```json
{
  "transaction_id": "txn_123",
  "feature_set_version": "fs_fraud_v1",
  "amount_usd": 1000.0
}
```

**Response**:
```json
{
  "transaction_id": "txn_123",
  "risk_score": 0.75,
  "decision": "MANUAL_REVIEW",
  "reason": "medium_risk_manual_review",
  "rule_applied": "medium_risk_manual_review",
  "request_id": "req_xxx",
  "model_version": "v1.0",
  "feature_set_version": "fs_fraud_v1",
  "latency_ms": 45
}
```

**SLO**: p95 ≤ 80ms (stricter than churn's 120ms)

### 2. POST `/score_async` - Asynchronous Batch Scoring
Submit a batch scoring task, returns job_id. Tasks are processed in the background by Celery workers.

**Request**:
```json
{
  "transaction_ids": ["txn_1", "txn_2", "txn_3"],
  "domain": "fraud",
  "callback_url": "https://example.com/callback"
}
```

**Response**:
```json
{
  "job_id": "job_xxx",
  "status": "queued",
  "created_at": "2026-01-26T10:00:00"
}
```

### 3. GET `/jobs/{job_id}` - Query Job Status
Query the status and results of an asynchronous task.

**Response** (success):
```json
{
  "job_id": "job_xxx",
  "domain": "fraud",
  "status": "succeeded",
  "payload": {...},
  "result": {
    "total": 3,
    "succeeded": 3,
    "failed": 0,
    "results": [
      {
        "transaction_id": "txn_1",
        "risk_score": 0.65,
        "decision": "APPROVE",
        "reason": "low_risk_approve"
      }
    ],
    "errors": []
  },
  "created_at": "2026-01-26T10:00:00",
  "updated_at": "2026-01-26T10:00:05"
}
```

### 4. GET `/health` - Health Check
Automatically added by platform SDK's `app_factory`.

### 5. GET `/metrics` - Prometheus Metrics
Automatically added by platform SDK's `metrics` module.

## Decision Types

- **APPROVE**: Low risk (risk_score < 0.5), auto-approve
- **REJECT**: High risk (risk_score ≥ 0.9), auto-reject
- **MANUAL_REVIEW**: Medium risk (0.5 ≤ risk_score < 0.9), manual review

## Policy Rules

The policy engine makes decisions based on the following rules (priority from high to low):

1. **high_risk_reject**: risk_score ≥ 0.9 → REJECT
2. **high_amount_manual_review**: risk_score ≥ 0.7 and amount_usd ≥ 10000 → MANUAL_REVIEW
3. **new_device_manual_review**: risk_score ≥ 0.6 and is_new_device == True → MANUAL_REVIEW
4. **country_change_manual_review**: risk_score ≥ 0.6 and country_change_flag_24h == True → MANUAL_REVIEW
5. **medium_risk_manual_review**: risk_score ≥ 0.5 → MANUAL_REVIEW
6. **low_risk_approve**: risk_score < 0.5 → APPROVE

## Async Task Processing

Async tasks are processed using Celery + Redis. You need to start a Celery worker:

```bash
# Start Celery worker
celery -A fraud.serving.async_tasks worker --loglevel=info
```

Tasks are defined in `async_tasks.py` using the platform SDK's `create_async_task` decorator.

## Audit Trail

All scoring requests are automatically written to the `platform.prediction_audit` table, including:
- request_id
- domain, model_name, model_version
- feature_set_version
- entity_key (transaction_id)
- latency_ms
- predictions (risk_score)
- decision (decision, reason, rule_applied)
- trace_id (if provided)

## Dependencies

- `platform_sdk`: Platform unified SDK
- `fraud.models`: Fraud risk model
- `fraud.policy`: Policy engine

## Configuration

Service uses platform SDK's unified configuration (via `platform_sdk.common.config.Config`):
- Database connection
- Redis/Celery broker
- MLflow tracking URI
- Logging configuration

## SLO (Stricter)

- **p95 latency**: ≤ 80ms (vs churn's 120ms)
- **p99 latency**: ≤ 150ms
- **5xx error rate**: ≤ 0.1% (vs churn's 0.5%)

## Differences from Project 1

| Feature | Project 1 (Churn) | Project 2 (Fraud) |
|------|------------------|-------------------|
| Entity | user_id | transaction_id |
| Output | churn_prob, ltv_90d | risk_score |
| Decision | NO_ACTION/OFFER_* | APPROVE/REJECT/MANUAL_REVIEW |
| SLO p95 | 120ms | **80ms** |
| SLO 5xx | 0.5% | **0.1%** |
| Core Component | Decisioning Optimizer | Policy Engine |
