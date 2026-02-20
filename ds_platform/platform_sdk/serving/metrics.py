"""
Prometheus Metrics Endpoint
"""

from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


def add_metrics_endpoint(app: FastAPI):
    """Add /metrics endpoint for Prometheus"""
    
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
