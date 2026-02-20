"""
FastAPI Serving Application for Fraud Risk Scoring
Provides /score (sync) and /score_async (async) endpoints
Uses platform_sdk for unified middleware and utilities
"""

import sys
from pathlib import Path

# project_2_fraud_risk_scoring root (for artifacts); workspace root = one level up (for ds_platform)
_project_root = Path(__file__).parent.parent.parent
_workspace_root = _project_root.parent
sys.path.insert(0, str(_workspace_root / "ds_platform"))

from fastapi import HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import uuid
import logging

from platform_sdk.serving.app_factory import create_app
from platform_sdk.serving.metrics import add_metrics_endpoint
from platform_sdk.schemas.audit import AuditRecord
from platform_sdk.db.audit_writer import write_audit_async
from platform_sdk.serving.async_queue import AsyncJobManager
from platform_sdk.feature_store.materialize import FeatureMaterializer

from fraud.serving.scoring import ScoringService

# Create app using platform factory
app = create_app(
    title="Fraud / Risk Scoring Service",
    version="1.0.0",
    domain="fraud"
)

# Add metrics endpoint
add_metrics_endpoint(app)

# Initialize scoring service (loads local model when USE_LOCAL_MODEL=1)
_default_model = _project_root / "artifacts" / "model.pkl"
_scoring_model_path = str(_default_model) if _default_model.exists() else None
scoring_service = ScoringService(model_path=_scoring_model_path)

# Initialize async job manager
job_manager = AsyncJobManager()
feature_materializer = FeatureMaterializer(domain="fraud")


class ScoreRequest(BaseModel):
    """Synchronous scoring request"""
    transaction_id: str
    feature_set_version: Optional[str] = "fs_fraud_v1"
    amount_usd: Optional[float] = None  # Optional, can be in features
    features: Optional[Dict[str, float]] = None  # Local mode: send feature dict (same keys as training)


class ScoreResponse(BaseModel):
    """Scoring response"""
    transaction_id: str
    risk_score: float
    decision: str  # APPROVE / REJECT / MANUAL_REVIEW
    reason: str
    rule_applied: Optional[str] = None
    request_id: str
    model_version: str
    feature_set_version: str
    latency_ms: Optional[int] = None


class AsyncScoreRequest(BaseModel):
    """Async scoring request"""
    transaction_ids: List[str]
    domain: str = "fraud"
    callback_url: Optional[str] = None


class AsyncJobResponse(BaseModel):
    """Async job response"""
    job_id: str
    status: str
    created_at: str


class OnlineFeatureUpsertRequest(BaseModel):
    """Upsert one entity into fraud.online_features"""
    entity_key: str
    feature_set_version: Optional[str] = "fs_fraud_v1"
    features: Dict[str, float]
    ttl_seconds: Optional[int] = None


class OnlineFeatureUpsertResponse(BaseModel):
    """Response after upserting online feature"""
    entity_key: str
    feature_set_version: str
    status: str = "ok"


# Health endpoint is added by create_app()


@app.get("/")
async def root():
    """Root: point to docs and main endpoints."""
    return {
        "service": "Fraud / Risk Scoring",
        "docs": "/docs",
        "score": "POST /score",
        "score_async": "POST /score_async",
        "admin": "POST /admin/online-features",
        "health": "/health",
    }


@app.post("/admin/online-features", response_model=OnlineFeatureUpsertResponse)
async def upsert_online_feature(body: OnlineFeatureUpsertRequest):
    """
    Upsert one entity's features into fraud.online_features (for testing / admin).
    """
    try:
        feature_materializer.upsert_one(
            entity_key=body.entity_key,
            feature_set_version=body.feature_set_version or "fs_fraud_v1",
            features=body.features,
            ttl_seconds=body.ttl_seconds,
        )
        return OnlineFeatureUpsertResponse(
            entity_key=body.entity_key,
            feature_set_version=body.feature_set_version or "fs_fraud_v1",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse)
async def score(body: ScoreRequest, request: Request):
    """
    Synchronous scoring endpoint
    
    Returns risk_score, decision (APPROVE/REJECT/MANUAL_REVIEW), and reason
    SLO: p95 ≤ 80ms (stricter than churn's 120ms)
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Score transaction
        result = scoring_service.score_transaction(
            transaction_id=body.transaction_id,
            feature_set_version=body.feature_set_version,
            amount_usd=body.amount_usd,
            features=body.features
        )
        
        # Compute latency
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Audit log using platform SDK
        audit_record = AuditRecord(
            request_id=request_id,
            domain="fraud",
            model_name="fraud_riskscore",
            model_version=result.get("model_version", "unknown"),
            feature_set_version=result.get("feature_set_version", body.feature_set_version),
            entity_key=body.transaction_id,
            latency_ms=latency_ms,
            predictions={"risk_score": result["risk_score"]},
            decision={
                "decision": result["decision"],
                "reason": result["reason"],
                "rule_applied": result.get("rule_applied")
            },
            trace_id=getattr(request.state, "trace_id", None)
        )
        # Write to database (async, non-blocking)
        try:
            await write_audit_async(audit_record)
        except Exception as e:
            logging.error(f"Failed to write audit record: {e}")
        
        return ScoreResponse(
            transaction_id=body.transaction_id,
            risk_score=result["risk_score"],
            decision=result["decision"],
            reason=result["reason"],
            rule_applied=result.get("rule_applied"),
            request_id=request_id,
            model_version=result.get("model_version", "unknown"),
            feature_set_version=result.get("feature_set_version", body.feature_set_version),
            latency_ms=latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_async", response_model=AsyncJobResponse)
async def score_async(request: AsyncScoreRequest):
    """
    Async scoring endpoint
    
    Enqueues job using Celery/Redis and returns job_id
    """
    # Prepare payload
    payload = {
        "transaction_ids": request.transaction_ids,
        "feature_set_version": "fs_fraud_v1"
    }
    
    # Enqueue job using platform SDK AsyncJobManager
    job_id = job_manager.enqueue_job(
        domain=request.domain,
        task_name="fraud.score_batch",  # Celery task name
        payload=payload,
        callback_url=request.callback_url
    )
    
    # Get job status
    job_status = job_manager.get_job_status(job_id)
    
    return AsyncJobResponse(
        job_id=job_id,
        status=job_status.get("status", "queued"),
        created_at=job_status.get("created_at", datetime.now().isoformat())
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get async job status
    
    Returns current status, result (if succeeded), or error (if failed)
    """
    job_status = job_manager.get_job_status(job_id)
    
    if job_status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port from churn (8000)
