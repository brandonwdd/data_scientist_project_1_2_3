"""
FastAPI App Factory
Creates FastAPI app with platform middleware
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

from platform_sdk.serving.middleware import RequestIDMiddleware, MetricsMiddleware
from platform_sdk.serving.health import add_health_endpoint


def create_app(
    title: str,
    version: str = "1.0.0",
    domain: str = "default",
    enable_cors: bool = True,
    enable_metrics: bool = True,
    **kwargs
) -> FastAPI:
    """
    Create FastAPI app with platform middleware
    
    Args:
        title: App title
        version: App version
        domain: Domain name (churn, fraud, rag)
        enable_cors: Enable CORS middleware
        enable_metrics: Enable Prometheus metrics
        **kwargs: Additional FastAPI kwargs
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(title=title, version=version, **kwargs)
    
    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Platform middleware
    app.add_middleware(RequestIDMiddleware)
    
    if enable_metrics:
        app.add_middleware(MetricsMiddleware, domain=domain)
    
    # Health endpoint
    add_health_endpoint(app, service_name=title, domain=domain)
    
    return app
