# Infra Docker Services

What services come up after `docker compose up -d`, their names, roles, architecture, and what you see inside Docker.

- **Project name**: `name: ds_platform` → top-level display in Docker as **ds_platform**
- **Container / volume names**: Unified as **infra-&lt;service&gt;** (see table below); independent of the three projects (projects connect via localhost:port only).

### Container / Image / Volume Mapping

| Service | Container name | Image | Volume name |
|---------|----------------|-------|-------------|
| postgres | **infra-postgres** | postgres:15-alpine | **infra-postgres** |
| redis | **infra-redis** | redis:7-alpine | **infra-redis** |
| mlflow | **infra-mlflow** | ghcr.io/mlflow/mlflow:v2.8.1 | **infra-mlflow** |
| prometheus | **infra-prometheus** | prom/prometheus:v2.45.0 | **infra-prometheus** |
| qdrant | **infra-qdrant** | qdrant/qdrant:v1.7.4 | **infra-qdrant** |

Note: Image names come from upstream (repo:tag) and are not renamed to infra-xxx; container and volume names are unified as infra-&lt;service&gt;. The three projects connect via **localhost + port** and do not depend on container/volume names; no code changes required.

---

## 1. Default 5 services (docker-compose.yml)

| # | Service | Container name | Image | Volume (docker volume ls) | Host port | Role |
|---|---------|----------------|-------|---------------------------|-----------|------|
| 1 | **postgres** | infra-postgres | postgres:15-alpine | infra-postgres | 5432 | Relational DB: churn / fraud / rag; init.sql creates schemas and tables |
| 2 | **redis** | infra-redis | redis:7-alpine | infra-redis | 6379 | Cache + Celery broker/result backend |
| 3 | **mlflow** | infra-mlflow | ghcr.io/mlflow/mlflow:v2.8.1 | infra-mlflow | 5000 | Experiment tracking + model registry; backend Postgres, artifacts on local volume |
| 4 | **prometheus** | infra-prometheus | prom/prometheus:v2.45.0 | infra-prometheus | 9090 | Scrapes /metrics from services, stores time-series data |
| 5 | **qdrant** | infra-qdrant | qdrant/qdrant:v1.7.4 | infra-qdrant | 6333, 6334 | Vector store for RAG (project_3_enterprise_rag_llm) |

---

## 2. Per-service details

### 1. postgres
- **Image**: `postgres:15-alpine`, built from `infra/postgres/Dockerfile`.
- **Env**: `POSTGRES_USER` (default mlplatform), `POSTGRES_PASSWORD` (default mlplatform_dev), `POSTGRES_DB` (default churn; only determines the DB created on first run).
- **Data**: Named volume **infra-postgres** mounted at `/var/lib/postgresql/data`; `./postgres/init.sql` mounted at `/docker-entrypoint-initdb.d/init.sql`, executed on first start.
- **init.sql**: Creates databases `fraud`, `rag` (churn is created by container env); creates schemas and tables per DB (e.g. churn.* / fraud.* / rag.*: prediction_audit, async_jobs, online_features, etc.).

### 2. redis
- **Image**: Official `redis:7-alpine`.
- **Command**: `redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis_dev}` (persistence + password).
- **Data**: Named volume **infra-redis** mounted at `/data`.

### 3. mlflow
- **Image**: `ghcr.io/mlflow/mlflow:v2.8.1`.
- **Depends on**: postgres healthy before start.
- **Env**: `MLFLOW_BACKEND_STORE_URI` points to postgres (default DB name is mlplatform; create it or change env if missing); `MLFLOW_ARTIFACT_ROOT=file:///mlflow/artifacts` uses local volume.
- **Data**: Named volume **infra-mlflow** at `/mlflow/artifacts`.
- **Command**: `mlflow server --backend-store-uri ... --default-artifact-root ... --host 0.0.0.0 --port 5000`.

### 4. prometheus
- **Image**: `prom/prometheus:v2.45.0`.
- **Config**: `./prometheus/prometheus.yml` mounted at `/etc/prometheus/prometheus.yml`; rules in `alert_rules.yml` via rule_files.
- **Data**: Named volume **infra-prometheus** for TSDB.
- **Current scrape targets**: self (localhost:9090), ds-platform-services (placeholder), rag-service (host.docker.internal:8002), mlflow (mlflow:5000).

### 5. qdrant
- **Image**: `qdrant/qdrant:v1.7.4`.
- **Ports**: 6333 HTTP API, 6334 gRPC.
- **Data**: Named volume **infra-qdrant** at `/qdrant/storage`.

---

## 3. Optional monitoring (docker-compose.monitoring.yml)

Usage:
```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

Adds 3 services:

| Service | Container name | Image | Port | Role |
|---------|----------------|-------|------|------|
| redis-exporter | infra-redis-exporter | oliver006/redis_exporter:v1.55.0 | 9121 | Expose Redis metrics to Prometheus |
| postgres-exporter | infra-postgres-exporter | quay.io/prometheuscommunity/postgres-exporter:v0.15.0 | 9187 | Expose Postgres metrics |
| grafana | infra-grafana | grafana/grafana:10.2.0 | 3000 | Dashboards; connects to Prometheus; admin/admin |

---

## 4. Architecture (inside Docker)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  ds-platform-network (bridge)             │
                    │                                                           │
  host (this machine)│   ┌──────────────┐     ┌──────────────┐                  │
  project_1_churn_ltv_decisioning/2/3 │   │  postgres    │     │    redis     │                  │
  connect via       │   │  :5432       │     │    :6379     │                  │
  localhost         │   │  (3 DBs)     │     │              │                  │
  :5432/:6379/...   │   └──────┬───────┘     └──────┬───────┘                  │
                    │          │                    │                          │
                    │          │ depends_on         │                          │
                    │          ▼                    │                          │
                    │   ┌──────────────┐            │                          │
                    │   │   mlflow     │            │                          │
                    │   │   :5000      │            │                          │
                    │   │ (backend=pg) │            │                          │
                    │   └──────────────┘            │                          │
                    │                               │                          │
                    │   ┌──────────────┐   ┌───────┴───────┐                  │
                    │   │  prometheus  │   │    qdrant      │                  │
                    │   │   :9090      │   │  :6333 :6334   │                  │
                    │   │ (scrape      │   │  (RAG vectors) │                  │
                    │   │  /metrics)   │   └────────────────┘                  │
                    │   └──────────────┘                                        │
                    └─────────────────────────────────────────────────────────┘
                                         │
                    Host port mapping: 5432, 6379, 5000, 9090, 6333, 6334
```

- All containers are on **ds-platform-network**; use service names to talk (e.g. mlflow → `postgres:5432`).
- Projects running on the host (project_1_churn_ltv_decisioning, project_2_fraud_risk_scoring, project_3_enterprise_rag_llm) reach services via **localhost:5432 / 6379 / 5000 / 6333**.
- Prometheus scrapes the RAG service on the host via `host.docker.internal:8002`.

---

## 5. What you see in Docker

### 1. Containers (docker ps / docker compose ps)

After `docker compose up -d` you should see 5 containers with the **container_name** values above:

- infra-postgres  
- infra-redis  
- infra-mlflow  
- infra-prometheus  
- infra-qdrant  

With monitoring enabled you also get:  
- infra-redis-exporter  
- infra-postgres-exporter  
- infra-grafana  

### 2. Network (docker network ls)

- One network created by compose, name like `ds_platform_ds-platform-network` (project name + network name); top-level project is **ds_platform**, driver bridge.
- `docker network inspect <network_name>` shows IPs and connections for all listed containers.

### 3. Volumes (docker volume ls)

Named volumes (fixed as infra-&lt;service&gt;, aligned with container names):

- **infra-postgres**  
- **infra-redis**  
- **infra-mlflow**  
- **infra-prometheus**  
- **infra-qdrant**  
- With monitoring: grafana uses compose default name (e.g. `ds_platform_grafana_data`).

This is the mapping of service names, container names, network, and volumes you see in Docker.
