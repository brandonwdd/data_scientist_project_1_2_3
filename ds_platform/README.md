# ML Platform

Shared Foundation (SDK + infra + common specifications)

## Directory Structure

```
ds_platform/
├── platform_sdk/          # Platform SDK (unified capabilities)
│   ├── common/            # Common utilities (config/logging/time/ids)
│   ├── db/                # Database (pg.py, models.py)
│   ├── schemas/          # API contracts (api_common.py, scoring.py, audit.py)
│   ├── serving/          # Service scaffolding (app_factory.py, middleware.py, metrics.py, async_queue.py, health.py)
│   ├── training/         # Training tools (mlflow_client.py, optuna_runner.py, promotion_gate.py, eval_metrics.py)
│   ├── feature_store/    # Feature store (spec.py, offline_join.py, materialize.py, online_lookup.py, drift_baseline.py)
│   ├── quality/          # Quality gates (ge_runner.py, pandera_runner.py, contracts.py)
│   └── observability/    # Observability & governance (drift.py, tracing_stub.py)
└── infra/                # Infrastructure configuration
    ├── docker-compose.yml
    ├── postgres/init.sql
    ├── prometheus/
    └── k8s/
```

## Platform Responsibilities

### Platform Responsibilities (Unified Capabilities)
- Training/evaluation/registration: Optuna + MLflow Tracking/Registry + Promotion Gate
- Feature specification: feature_spec.yaml + offline/online materialization (Spark wrapper)
- Service scaffolding: FastAPI (sync + async) + middleware (request_id/log/metrics)
- Quality gates: GE/Pandera runner
- Observability & governance: Prometheus metrics + drift helpers (PSI/KS/ECE, etc.)
- Audit trail: Unified audit schema + database write utilities

### Platform Not Responsible (Business-Specific)
- Label definitions, business feature meanings, strategy backtest logic
- Use case model structures/thresholds/evaluation sets (RAG eval set)

## Usage

### Install SDK

```bash
cd ds_platform/platform_sdk
pip install -e .
```

### Use in Projects

```python
from platform_sdk.serving import create_app
from platform_sdk.training import MLflowClient
from platform_sdk.feature_store import OnlineFeatureLookup

# Use platform SDK
app = create_app(domain="churn")
mlflow_client = MLflowClient()
feature_store = OnlineFeatureLookup(domain="churn")
```

## Reference Documentation

- [Infra Documentation](infra/README.md)
- [Platform SDK API Documentation](platform_sdk/README.md)
