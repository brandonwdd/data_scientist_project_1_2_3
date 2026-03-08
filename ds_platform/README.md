# DS Platform

Shared Foundation (SDK + infra + common specifications)

## Directory Structure (training vs online serving)

`ds_platform` provides shared **SDK + infra** used by all three projects. The table below summarizes which parts are mainly used for **training**, which are for **serving/infra**, and which are shared:

- **TRAIN**: model training / evaluation / registration / feature engineering  
- **SERVE**: online services, async jobs, infra, observability  
- **BOTH**: shared utilities and contracts

| Category | Path | Usage | Notes |
|----------|------|-------|-------|
| SDK root | `platform_sdk/` | BOTH | Core platform SDK used by all projects |
| SDK | `platform_sdk/common/` | BOTH | Shared utilities (config, logging, time, IDs) |
| SDK | `platform_sdk/db/` | BOTH | Database helpers (`pg.py`, ORM models, connections) |
| SDK | `platform_sdk/schemas/` | BOTH | API and DB contracts (`api_common.py`, `scoring.py`, `audit.py`, etc.) |
| SDK | `platform_sdk/serving/` | SERVE | Service scaffolding (`app_factory.py`, middleware, metrics, async_queue, health checks) |
| SDK | `platform_sdk/training/` | TRAIN | Training tools (MLflow client, Optuna runner, promotion gate, eval metrics) |
| SDK | `platform_sdk/feature_store/` | BOTH | Feature store (spec, offline join, materialize, online lookup, drift baselines) |
| SDK | `platform_sdk/quality/` | TRAIN | Data quality gates (Great Expectations / Pandera runners, contracts) |
| SDK | `platform_sdk/observability/` | SERVE | Observability & governance (drift helpers, tracing stubs, metrics wiring) |
| Infra root | `infra/` | SERVE | Shared infra configuration for all projects |
| Infra | `infra/docker-compose.yml` | SERVE | Spins up Postgres, Redis, MLflow, Prometheus, Qdrant, etc. |
| Infra | `infra/postgres/init.sql` | SERVE | Creates shared DBs/schemas (`churn`, `fraud`, `rag`, audit tables, async jobs, etc.) |
| Infra | `infra/prometheus/` | SERVE | Prometheus configuration for scraping metrics |
| Infra | `infra/k8s/` | SERVE | K8s manifests/templates for deploying services |

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
