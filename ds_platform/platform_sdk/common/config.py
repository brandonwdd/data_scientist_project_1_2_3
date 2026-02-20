"""
Configuration Management
Loads configuration from environment variables and config files
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path


class Config:
    """Platform configuration"""

    # Database
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "mlplatform")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "mlplatform_dev")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "churn")
    POSTGRES_SCHEMA: str = os.getenv("POSTGRES_SCHEMA", "churn")  # churn / fraud / rag; must match DB name

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "redis_dev")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_BACKEND_STORE_URI: str = os.getenv(
        "MLFLOW_BACKEND_STORE_URI",
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    MLFLOW_ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "file:///mlflow/artifacts")

    # Feature Store
    FEATURE_STORE_OFFLINE_PATH: str = os.getenv("FEATURE_STORE_OFFLINE_PATH", "s3://lake/features")
    FEATURE_STORE_ONLINE_TTL_HOURS: int = int(os.getenv("FEATURE_STORE_ONLINE_TTL_HOURS", "24"))

    # Qdrant (RAG vector store)
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # Async Queue
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL",
        f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND",
        f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        return (
            f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )

    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL"""
        return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
