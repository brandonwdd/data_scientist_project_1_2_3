"""
FastAPI Serving Application
Provides /score (sync) and /score_async (async) endpoints
Uses platform_sdk for unified middleware and utilities
"""

import sys
from pathlib import Path

# Add ds_platform so "from platform_sdk.xxx" resolves
_ws_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_ws_root / "ds_platform") not in sys.path:
    sys.path.insert(0, str(_ws_root / "ds_platform"))

from fastapi import HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import os
import uuid
import logging
import pandas as pd

from platform_sdk.serving.app_factory import create_app
from platform_sdk.feature_store.materialize import FeatureMaterializer
from platform_sdk.serving.metrics import add_metrics_endpoint
from platform_sdk.schemas.audit import AuditRecord
from platform_sdk.db.audit_writer import write_audit_async
from platform_sdk.serving.async_queue import AsyncJobManager

from churn.serving.scoring import ScoringService

# Create app using platform factory
app = create_app(
    title="Churn + LTV Decisioning Service",
    version="1.0.0",
    domain="churn"
)

# Add metrics endpoint
add_metrics_endpoint(app)

# Initialize scoring service (local: set CHURN_MODEL_PATH, LTV_MODEL_PATH)
_churn_path = os.environ.get("CHURN_MODEL_PATH")
_ltv_path = os.environ.get("LTV_MODEL_PATH")
scoring_service = ScoringService(
    churn_model_path=_churn_path,
    ltv_model_path=_ltv_path,
)

# Initialize async job manager
job_manager = AsyncJobManager()


class ScoreRequest(BaseModel):
    """Synchronous scoring request"""
    user_id: str
    feature_set_version: Optional[str] = "fs_churn_v1"


class ScoreResponse(BaseModel):
    """Scoring response"""
    user_id: str
    churn_prob: float
    ltv_90d: float
    action: str
    reason_codes: List[str]
    request_id: str
    model_version: str
    feature_set_version: str


class AsyncScoreRequest(BaseModel):
    """Async scoring request"""
    user_ids: List[str]
    domain: str = "churn"
    callback_url: Optional[str] = None


class AsyncJobResponse(BaseModel):
    """Async job response"""
    job_id: str
    status: str
    created_at: str


class ExplainRequest(BaseModel):
    """Explainability request"""
    user_id: str
    feature_set_version: Optional[str] = "fs_churn_v1"


class ExplainResponse(BaseModel):
    """Explainability response"""
    user_id: str
    reason_codes: List[str]
    top_features: List[Dict[str, float]]
    shap_values: Dict[str, float]


class MaterializeRequest(BaseModel):
    """Materialize features to online store (churn.online_features)"""
    user_ids: List[str]
    feature_set_version: Optional[str] = "fs_churn_v1"


class MaterializeResponse(BaseModel):
    """Materialize response"""
    materialized: int
    feature_set_version: str
    message: str


# Health endpoint is added by create_app()


@app.post("/score", response_model=ScoreResponse)
async def score(request: Request, score_request: ScoreRequest):
    """
    Synchronous scoring endpoint

    Returns churn_prob, ltv_90d, action, and reason_codes
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()

    try:
        result = scoring_service.score_user(
            user_id=score_request.user_id,
            feature_set_version=score_request.feature_set_version
        )

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        audit_record = AuditRecord(
            request_id=request_id,
            domain="churn",
            model_name="churn_churnrisk",
            model_version=result.get("model_version", "unknown"),
            feature_set_version=result.get("feature_set_version", score_request.feature_set_version),
            entity_key=score_request.user_id,
            latency_ms=latency_ms,
            predictions={"churn_prob": result["churn_prob"], "ltv_90d": result["ltv_90d"]},
            decision={"action": result["action"], "reason_codes": result["reason_codes"]},
            trace_id=getattr(request.state, "trace_id", None)
        )
        try:
            await write_audit_async(audit_record)
        except Exception as e:
            logging.error(f"Failed to write audit record: {e}")

        return ScoreResponse(
            user_id=score_request.user_id,
            churn_prob=result["churn_prob"],
            ltv_90d=result["ltv_90d"],
            action=result["action"],
            reason_codes=result["reason_codes"],
            request_id=request_id,
            model_version=result.get("model_version", "unknown"),
            feature_set_version=result.get("feature_set_version", score_request.feature_set_version)
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
        "user_ids": request.user_ids,
        "feature_set_version": "fs_churn_v1"  # Could be configurable
    }
    
    # Enqueue job using platform SDK AsyncJobManager
    job_id = job_manager.enqueue_job(
        domain=request.domain,
        task_name="churn.score_batch",  # Celery task name
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


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Explainability endpoint
    
    Returns reason codes and top SHAP features
    """
    try:
        explanation = scoring_service.explain_user(
            user_id=request.user_id,
            feature_set_version=request.feature_set_version
        )
        
        return ExplainResponse(**explanation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/admin/materialize", response_model=MaterializeResponse)
async def materialize_features(request: MaterializeRequest):
    """
    Materialize features to churn.online_features (production-like).
    Builds feature rows for given user_ids and writes to online store.
    """
    try:
        rows = []
        for user_id in request.user_ids:
            features = scoring_service._mock_features_for_user(user_id)
            row = {"user": user_id, **features}
            rows.append(row)
        if not rows:
            return MaterializeResponse(
                materialized=0,
                feature_set_version=request.feature_set_version or "fs_churn_v1",
                message="No user_ids"
            )
        features_df = pd.DataFrame(rows)
        materializer = FeatureMaterializer(domain="churn")
        materializer.materialize_features(
            features_df,
            feature_set_version=request.feature_set_version or "fs_churn_v1",
            entity_col="user"
        )
        return MaterializeResponse(
            materialized=len(features_df),
            feature_set_version=request.feature_set_version or "fs_churn_v1",
            message=f"Materialized {len(features_df)} entities to churn.online_features"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
