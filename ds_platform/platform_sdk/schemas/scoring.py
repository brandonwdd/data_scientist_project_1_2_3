"""Scoring API Schemas: ScoreRequest, ScoreResponse, AsyncScoreRequest, AsyncJobResponse"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from platform_sdk.schemas.api_common import RequestMeta


class ScoreRequest(BaseModel):
    """Synchronous scoring request"""
    entity_key: str  # e.g., user_id
    feature_set_version: Optional[str] = None
    meta: Optional[RequestMeta] = None


class ScoreResponse(BaseModel):
    """Scoring response"""
    entity_key: str
    predictions: Dict[str, Any]  # Model predictions
    decision: Optional[Dict[str, Any]] = None  # Business decision
    request_id: str
    model_version: str
    feature_set_version: str
    latency_ms: Optional[int] = None


class AsyncScoreRequest(BaseModel):
    """Async scoring request"""
    entity_keys: List[str]
    domain: str
    callback_url: Optional[str] = None
    meta: Optional[RequestMeta] = None


class AsyncJobResponse(BaseModel):
    """Async job response"""
    job_id: str
    status: str  # queued, running, succeeded, failed
    created_at: str
    domain: str
