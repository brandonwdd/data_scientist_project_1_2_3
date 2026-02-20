"""
FastAPI Middleware
Request ID, Logging, Metrics
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from platform_sdk.common.logging import setup_logging
from platform_sdk.common.ids import generate_request_id, generate_trace_id

logger = setup_logging(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request_id to request and response"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request_id
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
        
        # Add to request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics middleware"""

    def __init__(self, app: ASGIApp, domain: str):
        super().__init__(app)
        self.domain = domain
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            from prometheus_client import Counter, Histogram
            
            self.request_count = Counter(
                f"http_requests_total",
                "Total HTTP requests",
                ["domain", "method", "endpoint", "status"]
            )
            
            self.request_duration = Histogram(
                f"http_request_duration_seconds",
                "HTTP request duration",
                ["domain", "method", "endpoint"]
            )
        except ImportError:
            logger.warning("prometheus_client not available, metrics disabled")
            self.request_count = None
            self.request_duration = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        if self.request_count and self.request_duration:
            duration = time.time() - start_time
            endpoint = request.url.path
            method = request.method
            status = response.status_code
            
            self.request_count.labels(
                domain=self.domain,
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            self.request_duration.labels(
                domain=self.domain,
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        return response
