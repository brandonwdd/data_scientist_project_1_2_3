"""
Common API Schemas
RequestMeta, ErrorResponse, HealthResponse
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class RequestMeta(BaseModel):
    """Request metadata"""
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    error_code: Optional[str] = None
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str
    version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
