# Project 1: Churn + LTV Decisioning Service

Subscription Churn and Lifetime Value Decisioning System

## Executive Summary

- Predict **30d churn_prob** + **90d LTV**
- Output **recommended_action** + **reason_codes** under budget constraints
- Support `/score` (synchronous) and `/score_async` (batch)
- Features: data quality gates, MLflow registration/auto-promotion, drift monitoring, audit trail

---

## Project Structure

```
project_1_churn_ltv_decisioning/
├── churn/
│   ├── configs/
│   │   ├── feature_spec.yaml          # Feature specification (windows, types, domain_sets)
│   │   └── promotion_gate.yaml       # Promotion gate thresholds (churn/ltv/decisioning)
│   ├── data/
│   │   ├── ravenstack_loader.py       # CSV load + churn/LTV label computation
│   │   ├── saas_churn_ltv/            # RavenStack CSVs
│   │   └── (moved) model artifacts now live in churn/models/artifacts/
│   ├── features/
│   │   ├── ravenstack_features.py     # Feature engineering for RavenStack (pandas)
│   │   ├── engineering.py             # Spark feature engineering (needs DataLoader)
│   │   └── point_in_time.py           # Point-in-time join (no future leakage)
│   ├── models/
│   │   ├── churn_model.py             # Churn classifier (LogReg/LightGBM + calibration)
│   │   ├── ltv_model.py               # LTV regression (LightGBM)
│   │   └── explainability.py         # SHAP reason codes
│   ├── decisioning/
│   │   ├── optimizer.py              # Budget-constrained optimizer (greedy/ILP)
│   │   └── backtest.py                # Backtest framework
│   ├── evaluation/
│   │   ├── metrics.py                 # AUC, PR-AUC, ECE, LTV metrics, decisioning metrics
│   │   └── protocol.py                # Rolling time-split evaluation
│   ├── training/
│   │   ├── train_churn_ravenstack.py   # Main training script (local CSV, no Spark)
│   │   ├── train_churn.py             # MLflow training (needs DataLoader)
│   │   ├── artifacts.py               # MLflow artifacts (metrics, plots, model card)
│   │   └── promotion_gate.py          # Gate evaluation (pr_auc_min, auc_min, ...)
│   ├── serving/
│   │   ├── app.py                     # FastAPI: /score, /score_async, /explain, /jobs
│   │   ├── scoring.py                 # ScoringService: features → models → optimizer
│   │   └── async_tasks.py             # Celery batch scoring task
│   ├── monitoring/
│   │   ├── drift_job.py               # Drift monitoring (PSI/KS)
│   │   └── rollback.py                # Rollback runbook (MLflow stage)
│   └── demo/
│       └── demo_5min.py               # 5-min demo workflow
├── requirements-local.txt             # Minimal deps (no Spark/S3)
└── README.md
```

---

## 文件链路 (File Flow)

### 1. 训练链路 (Training Pipeline)

本地训练使用 **RavenStack CSV**，入口为 `train_churn_ravenstack.py`，调用关系如下：

```
train_churn_ravenstack.py
    │
    ├─► churn/training/ravenstack_loader.py
    │       • load_ravenstack_tables()     → 5 张 CSV → accounts, subscriptions, churn_events, feature_usage, support_tickets
    │       • get_churn_labels()          → 30 天流失标签 (user_id, as_of_time, churn)
    │       • get_ltv_labels()            → 90 天 LTV 标签 (user_id, as_of_time, ltv_90d)
    │
    ├─► churn/features/ravenstack_features.py
    │       • compute_ravenstack_features(tables, as_of_times)  → 13 维特征 (FEATURE_NAMES)
    │       • 严格按 as_of_time 截断，无未来信息
    │
    ├─► churn/models/churn_model.py
    │       • ChurnModel(model_type="lightgbm", calibrate=True)
    │       • train() → predict_proba()
    │
    ├─► churn/models/ltv_model.py
    │       • LTVModel(model_type="lightgbm")
    │       • train() → predict() → ltv_90d
    │
    ├─► churn/evaluation/metrics.py
    │       • compute_churn_metrics()     → auc, pr_auc, ece, brier, topk_precision, lift
    │       • compute_ltv_metrics()       → mae, rmse, smape
    │
    └─► 输出
            • churn/models/artifacts/churn_model.pkl
            • churn/models/artifacts/ltv_model.pkl
```

**配置与门禁（可选）：**

- `churn/configs/feature_spec.yaml`：特征名、窗口、类型（engineering 路径会读）
- `churn/configs/promotion_gate.yaml`：上线阈值（`train_churn.py` + platform_sdk PromotionGate 使用）
- `churn/training/promotion_gate.py`：本地门禁判断
- `churn/training/artifacts.py`：MLflow 产出物（与 `train_churn.py` 配合）

---

### 2. 服务链路 (Serving Pipeline)

请求从 **FastAPI** 进入，经 **ScoringService** 完成特征 → 模型 → 决策 → 审计：

```
HTTP POST /score  (churn/serving/app.py)
    │
    ├─► churn/serving/scoring.py  →  ScoringService.score_user(user_id)
    │       │
    │       ├─► 特征
    │       │       • platform_sdk.feature_store.online_lookup  → 在线特征 (entity_key=user:xxx)
    │       │       • 若无则 _mock_features_for_user(user_id)    → 本地/演示用 mock
    │       │
    │       ├─► churn/models/churn_model.py   →  ChurnModel.predict_proba(X)  → churn_prob
    │       ├─► churn/models/ltv_model.py     →  LTVModel.predict(X)          → ltv_90d
    │       │
    │       └─► churn/decisioning/optimizer.py
    │               • DecisionOptimizer.greedy_optimize(users_df, churn_probs, ltvs)
    │               • 输出 action (NO_ACTION / OFFER_5 / OFFER_10 / CALL_SUPPORT) + reason_codes
    │
    ├─► platform_sdk.db.audit_writer  →  write_audit_async(AuditRecord)  → churn.prediction_audit
    │
    └─► Response: churn_prob, ltv_90d, action, reason_codes, request_id, model_version, feature_set_version
```

**异步批量打分：**

```
POST /score_async  →  app.py  →  platform_sdk AsyncJobManager.enqueue_job()
    →  Celery 执行  churn/serving/async_tasks.py  →  score_batch_task()
            →  ScoringService.score_user()  per user
            →  write_audit()  per prediction
            →  结果写入 job result
GET /jobs/{job_id}  →  job_manager.get_job_status(job_id)
```

**可解释性：**

```
POST /explain  →  app.py  →  ScoringService.explain_user()
    →  _load_user_features() + score_user()  →  reason_codes
    →  简化 SHAP/特征重要性  →  top_features, shap_values
```

---

### 3. 链路小结表

| 阶段     | 入口/核心文件 | 依赖文件 |
|----------|----------------|----------|
| 训练入口 | `training/train_churn_ravenstack.py` | `data/ravenstack_loader`, `features/ravenstack_features`, `models/churn_model`, `models/ltv_model`, `evaluation/metrics` |
| 服务入口 | `serving/app.py` | `serving/scoring` → `models/*`, `decisioning/optimizer`, platform_sdk (audit, feature lookup, async_queue) |
| 决策逻辑 | `decisioning/optimizer.py` | 无其他 churn 模块依赖 |
| 回测     | `decisioning/backtest.py` | `decisioning/optimizer`, `evaluation/metrics`, platform_sdk logging |
| 监控     | `monitoring/drift_job.py`, `monitoring/rollback.py` | platform_sdk observability, MLflow |

---

## Quick Start

### Local run (RavenStack data, no S3/Spark)

1. **Enter project and install dependencies**

```bash
cd project_1_churn_ltv_decisioning
pip install -r requirements-local.txt
pip install -e ../ds_platform/platform_sdk
```

   For full environment (MLflow, Spark, etc.): `pip install -r requirements.txt`

2. **Train Churn + LTV** (data in `churn/data/saas_churn_ltv/`)

```bash
python churn/training/train_churn_ravenstack.py --as-of-date 2024-07-01
```

3. **Start the service**

```powershell
# PowerShell
$env:CHURN_MODEL_PATH="$PWD\churn\\models\\artifacts\churn_model.pkl"
$env:LTV_MODEL_PATH="$PWD\churn\\models\\artifacts\ltv_model.pkl"
$env:POSTGRES_DB="churn"
$env:POSTGRES_SCHEMA="churn"
python -m uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000
```

```bash
# Windows CMD
set CHURN_MODEL_PATH=%cd%\churn\\models\\artifacts\churn_model.pkl
set LTV_MODEL_PATH=%cd%\churn\\models\\artifacts\ltv_model.pkl
set POSTGRES_DB=churn
set POSTGRES_SCHEMA=churn
uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000
```

4. **Call /score**

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"user_id\": \"A-2e4581\"}"
```

---

## Local PostgreSQL (audit persistence)

Each `/score` call is written to **PostgreSQL** (database **churn**, schema **churn**, table `churn.prediction_audit`) via platform_sdk.

1. **Start Postgres** (from workspace root):

```bash
cd ds_platform/infra
docker compose up -d postgres
```

2. **Environment** (optional): `POSTGRES_DB=churn`, `POSTGRES_SCHEMA=churn`. Table schema is created by `init.sql`.

3. **Inspect audit** (e.g. in pgAdmin, connect to churn DB):

```sql
SELECT request_id, entity_key, predictions, decision, created_at
FROM churn.prediction_audit
ORDER BY created_at DESC
LIMIT 10;
```

---

## Async scoring (Celery)

For `/score_async`, run a Celery worker:

```bash
celery -A churn.serving.async_tasks worker --loglevel=info
```

Task name: `churn.score_batch`; payload: `user_ids`, `feature_set_version`.

---

## 5-Min Demo

1. Start infra: `cd ds_platform/infra && docker compose up -d`
2. Train: `python churn/training/train_churn_ravenstack.py --as-of-date 2024-07-01`
3. MLflow (if used): http://localhost:5000
4. Start API: `uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000`
5. Test: `POST /score` with `{"user_id": "A-2e4581"}`
6. Metrics: http://localhost:9090 (Prometheus)

---

## Document Sections (B1–B13)

- **B1–B2**: Scope & non-goals  
- **B3**: Time semantics & label definition  
- **B4**: Data & contracts  
- **B5**: Feature store (offline + online)  
- **B6**: Modeling  
- **B7**: Decisioning layer  
- **B8**: Offline evaluation protocol  
- **B9**: Serving & APIs  
- **B10**: Promotion gate  
- **B11**: Drift, retrain & rollback  
- **B12**: MLflow artifacts  
- **B13**: 5-min demo  
