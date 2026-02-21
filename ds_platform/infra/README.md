# ML Platform Infrastructure

This directory contains the shared infrastructure for all example ML projects in this repo. It provides a consistent, batteries‑included environment for:
- **Local development** using Docker Compose (Postgres, Redis, MLflow, Prometheus, Qdrant, optional Grafana)
- **Monitoring and observability** (Prometheus, alert rules, Grafana dashboards, Redis/Postgres exporters)
- **Production‑like deployment** templates on Kubernetes (base services + per‑project apps)

## Directory Structure

```
infra/
├── docker-compose.yml          # Local development environment
├── docker-compose.monitoring.yml  # Optional monitoring extension (Redis/Postgres exporter + Grafana)
├── env.development.example     # Development environment variable template
├── env.production.example      # Production environment variable template
├── postgres/
│   └── init.sql               # Database initialization script
├── prometheus/
│   ├── prometheus.yml         # Prometheus monitoring configuration
│   └── alert_rules.yml        # Prometheus alert rules
├── grafana/                   # Grafana configuration (optional)
│   └── provisioning/
│       ├── datasources/
│       └── dashboards/
├── scripts/                   # Operations scripts
│   ├── start.ps1             # Start script
│   ├── stop.ps1              # Stop script
│   ├── health-check.ps1      # Health check
│   ├── backup-db.ps1         # Database backup
│   ├── restore-db.ps1        # Database restore
│   └── cleanup.ps1           # Cleanup script
└── k8s/
    ├── base/                  # K8s base resources
    │   ├── namespace.yaml
    │   ├── configmap.yaml
    │   ├── secret.yaml.template
    │   ├── postgres-service.yaml
    │   ├── postgres-init-configmap.yaml
    │   ├── redis-service.yaml
    │   ├── mlflow-service.yaml
    │   └── prometheus-service.yaml
    └── apps/                  # Application deployment configuration
        ├── churn-service.yaml
        ├── fraud-service.yaml
        ├── rag-service.yaml
        └── ingress.yaml
```

## Local Development Environment (Docker Compose)

### Prerequisites

- Docker Desktop (Windows)
- Docker Compose v3.8+

### Quick Start

**Method 1: Using Script (Recommended)**
```powershell
cd infra
.\scripts\start.ps1
```

**Method 2: Manual Start**
1. **Copy environment variable file**
   ```powershell
   cd infra
   Copy-Item env.development.example .env
   ```

2. **Start all services**
   ```powershell
   docker compose up -d
   ```

3. **Verify service status**
   ```bash
   docker compose ps
   ```

4. **View logs**
   ```bash
   docker compose logs -f [service_name]
   ```

### Service Access Addresses

- **PostgreSQL**: `localhost:5432`
  - User: `dsplatform`
  - Password: `dsplatform_dev`
  - **Only three databases**: `churn` / `fraud` / `rag`, one per project; no other DBs
  - project_1_churn_ltv_decisioning uses `POSTGRES_DB=churn`, project_2_fraud_risk_scoring uses `POSTGRES_DB=fraud`, project_3_enterprise_rag_llm uses `POSTGRES_DB=rag`
  - In pgAdmin you will only see these three databases in the tree

- **Redis**: `localhost:6379`
  - Password: `redis_dev`

- **MLflow UI**: http://localhost:5000

- **Prometheus**: http://localhost:9090

### Stop Services

**Using script:**
```powershell
.\scripts\stop.ps1
```

**Or manually:**
```powershell
docker compose down
```

### Health Check

```powershell
.\scripts\health-check.ps1
```

### Database Backup & Restore

**Backup:**
```powershell
.\scripts\backup-db.ps1
# Or specify output path and filename
.\scripts\backup-db.ps1 -OutputPath "my-backups" -BackupName "backup.sql"
```

**Restore:**
```powershell
.\scripts\restore-db.ps1 -BackupFile "backups\dsplatform_20240101_120000.sql"
```

### Optional Monitoring Extension

Start Redis/Postgres exporter and Grafana:
```powershell
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

Access Grafana: http://localhost:3000
- Default username: `admin`
- Default password: `admin`

### Clean Data (Use with Caution)

```bash
docker compose down -v
```

## Production Environment (Kubernetes)

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configuration
- Access to S3 storage (for MLflow artifacts)

### Deployment Steps

1. **Create namespace and base resources**
   ```bash
   kubectl apply -f k8s/base/namespace.yaml
   kubectl apply -f k8s/base/configmap.yaml
   ```

2. **Create Secret**
   ```bash
   # Copy template and fill in actual values
   cp k8s/base/secret.yaml.template k8s/base/secret.yaml
   # Edit secret.yaml, fill in actual passwords and keys
   kubectl apply -f k8s/base/secret.yaml
   ```

3. **Create Postgres Init Script ConfigMap**
   ```bash
   # Create ConfigMap from init.sql file
   kubectl create configmap postgres-init-script \
     --from-file=init.sql=postgres/init.sql \
     --namespace=ds-platform \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

4. **Deploy base services**
   ```bash
   kubectl apply -f k8s/base/postgres-service.yaml
   kubectl apply -f k8s/base/redis-service.yaml
   kubectl apply -f k8s/base/mlflow-service.yaml
   kubectl apply -f k8s/base/prometheus-service.yaml
   ```

5. **Deploy application services**
   ```bash
   kubectl apply -f k8s/apps/churn-service.yaml
   kubectl apply -f k8s/apps/fraud-service.yaml
   kubectl apply -f k8s/apps/rag-service.yaml
   ```

6. **Deploy Ingress (optional)**
   ```bash
   kubectl apply -f k8s/apps/ingress.yaml
   ```

### Verify Deployment

```bash
# View all resources
kubectl get all -n ds-platform

# View service status
kubectl get pods -n ds-platform

# View logs
kubectl logs -f <pod-name> -n ds-platform
```

## Database Schema

The database initialization script `postgres/init.sql` creates **only three databases**: **churn / fraud / rag** (fully isolated):
- **churn**: schema **churn** only; tables churn.prediction_audit, churn.async_jobs, churn.online_features
- **fraud**: schema **fraud** only; tables fraud.prediction_audit, fraud.async_jobs, fraud.online_features
- **rag**: schema **rag** only; tables rag.prediction_audit, rag.async_jobs, rag.online_features, rag.feedback

### Per-project connection

| Project   | POSTGRES_DB | POSTGRES_SCHEMA | Table location |
|----------|-------------|-----------------|----------------|
| project_1_churn_ltv_decisioning (Churn) | `churn`  | `churn` | churn.prediction_audit etc. |
| project_2_fraud_risk_scoring (Fraud)  | `fraud`  | `fraud` | fraud.prediction_audit etc. |
| project_3_enterprise_rag_llm (RAG)   | `rag`    | `rag`   | rag.prediction_audit, rag.feedback etc. |

## Monitoring

### Prometheus

Prometheus configuration is in `prometheus/prometheus.yml`, default scrape interval 15 seconds.

**Alert Rules**: `prometheus/alert_rules.yml` includes the following alerts:
- Service availability (ServiceDown)
- API latency (HighLatency, CriticalLatency)
- Error rate (HighErrorRate, CriticalErrorRate)
- Database/Redis connection failures
- Async task failure rate
- SLO violations (latency, error rate)

### Application Metrics

All services should expose `/metrics` endpoint, including:
- HTTP request latency (histogram)
- QPS (counter)
- Error rate (counter)
- Business metrics (e.g., prediction latency, feature fetch latency, etc.)

### Grafana (Optional)

If monitoring extension is enabled, you can access Grafana to view pre-configured dashboards:
- http://localhost:3000
- Data source automatically configured as Prometheus

## Security Considerations

1. **Production passwords**: Must change default passwords in `.env.production` and `secret.yaml`
2. **Secret management**: Do not commit `secret.yaml` with real passwords to Git
3. **Network policies**: Production environments should configure NetworkPolicy to restrict Pod-to-Pod communication
4. **TLS certificates**: Production environments must enable HTTPS (via Ingress configuration)

## Troubleshooting

### PostgreSQL Connection Failure

```bash
# Check container status
docker compose ps postgres

# View logs
docker compose logs postgres

# Test connection
docker compose exec postgres psql -U dsplatform -d churn
```

### Redis Connection Failure

```bash
# Test connection
docker compose exec redis redis-cli -a redis_dev ping
```

### MLflow Not Accessible

```bash
# Check backend storage connection
docker compose logs mlflow

# Verify database connection
docker compose exec mlflow python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

## Maintenance

### Database Backup

**Local development (using script):**
```powershell
.\scripts\backup-db.ps1
```

**Manual backup:**
```powershell
docker compose exec postgres pg_dump -U dsplatform churn > backup_churn.sql
```

**Production environment:**
```bash
kubectl exec -n ds-platform <postgres-pod> -- pg_dump -U dsplatform_prod dsplatform > backup.sql
```

### Clean Resources

```powershell
# Stop containers only
.\scripts\cleanup.ps1

# Stop and delete volumes (data)
.\scripts\cleanup.ps1 -Volumes

# Stop and delete images
.\scripts\cleanup.ps1 -Images

# Full cleanup (containers+volumes+images)
.\scripts\cleanup.ps1 -All
```

### Upgrade Services

1. Update image versions
2. Rolling update deployment
3. Verify health checks pass

## Reference Documentation

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
