"""
Health Check Endpoint
"""

from fastapi import FastAPI
from datetime import datetime
from platform_sdk.schemas.api_common import HealthResponse


def add_health_endpoint(app: FastAPI, service_name: str, domain: str):
    """Add health check endpoint to app"""
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            service=service_name,
            version="1.0.0"
        )
