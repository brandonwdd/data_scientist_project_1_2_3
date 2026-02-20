# Platform SDK

ML Platform Unified Capabilities SDK

## Module Overview

### common/
Common utilities: configuration management, logging, time handling, ID generation

### db/
Database connection and model definitions

### schemas/
Unified API contracts (Pydantic models)

### serving/
Service scaffolding:
- `app_factory.py`: FastAPI application factory
- `middleware.py`: Request middleware (request_id, logging, metrics)
- `metrics.py`: Prometheus metrics
- `async_queue.py`: Async queue (Celery)
- `health.py`: Health check

### training/
Training tools:
- `mlflow_client.py`: MLflow client wrapper
- `optuna_runner.py`: Optuna hyperparameter optimization
- `promotion_gate.py`: Promotion gate framework
- `eval_metrics.py`: Evaluation metrics

### feature_store/
Feature store:
- `spec.py`: Feature specification parsing
- `offline_join.py`: Offline feature join
- `materialize.py`: Feature materialization
- `online_lookup.py`: Online feature lookup
- `drift_baseline.py`: Drift baseline

### quality/
Quality gates:
- `ge_runner.py`: Great Expectations runner
- `pandera_runner.py`: Pandera schema validation
- `contracts.py`: Data contracts

### observability/
Observability & governance:
- `drift.py`: Drift detection (PSI/KS/ECE)
- `tracing_stub.py`: Distributed tracing

## Installation

```bash
pip install -e .
```

## Usage Examples

See docstrings and example code in each module.
