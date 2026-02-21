"""Audit Schema: Unified AuditRecord for platform.prediction_audit"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """Unified audit record schema"""
    request_id: str
    domain: str
    model_name: str
    model_version: Optional[str] = None
    feature_set_version: Optional[str] = None
    feature_snapshot_hash: Optional[str] = None
    entity_key: Optional[str] = None
    latency_ms: Optional[int] = None
    predictions: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    trace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
