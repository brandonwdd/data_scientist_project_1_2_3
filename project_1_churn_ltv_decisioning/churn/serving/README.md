# Churn Serving Service

## Overview

Churn + LTV Decisioning Service provides synchronous and asynchronous scoring endpoints using the platform SDK's unified capabilities.

## API Endpoints

### 1. POST `/score` - Synchronous Scoring
Score a single user synchronously, returns churn_prob, ltv_90d, action, and reason_codes.

**Request**:
```json
{
  "user_id": "user_123",
  "feature_set_version": "fs_churn_v1"
}
```

**Response**:
```json
{
  "user_id": "user_123",
  "churn_prob": 0.65,
  "ltv_90d": 150.0,
  "action": "OFFER_10",
  "reason_codes": ["HIGH_CHURN_RISK", "HIGH_VALUE"],
  "request_id": "req_xxx",
  "model_version": "v1.0",
  "feature_set_version": "fs_churn_v1"
}
```

### 2. POST `/score_async` - Asynchronous Batch Scoring
Submit a batch scoring task, returns job_id. Tasks are processed in the background by Celery workers.

**Request**:
```json
{
  "user_ids": ["user_1", "user_2", "user_3"],
  "domain": "churn",
  "callback_url": "https://example.com/callback"  // optional
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
  "domain": "churn",
  "status": "succeeded",
  "payload": {...},
  "result": {
    "total": 3,
    "succeeded": 3,
    "failed": 0,
    "results": [...],
    "errors": []
  },
  "created_at": "2026-01-26T10:00:00",
  "updated_at": "2026-01-26T10:00:05"
}
```

**Response** (failure):
```json
{
  "job_id": "job_xxx",
  "status": "failed",
  "error": "Error message",
  ...
}
```

### 4. POST `/explain` - Explainability
Returns user's reason codes and top SHAP features.

**Request**:
```json
{
  "user_id": "user_123",
  "feature_set_version": "fs_churn_v1"
}
```

**Response**:
```json
{
  "user_id": "user_123",
  "reason_codes": ["HIGH_CHURN_RISK"],
  "top_features": [
    {"feature": "payment_fail_30d", "importance": 0.25},
    {"feature": "recency_days", "importance": 0.20}
  ],
  "shap_values": {
    "payment_fail_30d": 0.25,
    "recency_days": 0.20
  }
}
```

### 5. GET `/health` - Health Check
Automatically added by platform SDK's `app_factory`.

### 6. GET `/metrics` - Prometheus Metrics
Automatically added by platform SDK's `metrics` module.

## Async Task Processing

Async tasks are processed using Celery + Redis. You need to start a Celery worker:

```bash
# Start Celery worker
celery -A churn.serving.async_tasks worker --loglevel=info
```

Tasks are defined in `async_tasks.py` using the platform SDK's `create_async_task` decorator.

## Audit Trail

All scoring requests are automatically written to the `platform.prediction_audit` table, including:
- request_id
- domain, model_name, model_version
- feature_set_version
- entity_key (user_id)
- latency_ms
- predictions (churn_prob, ltv_90d)
- decision (action, reason_codes)
- trace_id (if provided)

## Dependencies

- `platform_sdk`: Platform unified SDK
- `churn.models`: Churn and LTV models
- `churn.decisioning`: Decision optimizer

## Configuration

Service uses platform SDK's unified configuration (via `platform_sdk.common.config.Config`):
- Database connection
- Redis/Celery broker
- MLflow tracking URI
- Logging configuration
