"""Database Models"""

import os
from typing import Optional
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
_SCHEMA = os.getenv("POSTGRES_SCHEMA", "churn")


class OnlineFeature(Base):
    """Model for {schema}.online_features"""
    __tablename__ = "online_features"
    __table_args__ = {"schema": _SCHEMA}

    domain = Column(String(50), primary_key=True)
    entity_key = Column(String(255), primary_key=True)
    feature_set_version = Column(String(50), primary_key=True)
    features = Column(JSON, nullable=False)
    event_time = Column(DateTime(timezone=True), nullable=False)
    materialized_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now)
    ttl_seconds = Column(Integer, nullable=True)


class PredictionAudit(Base):
    """Model for {schema}.prediction_audit"""
    __tablename__ = "prediction_audit"
    __table_args__ = {"schema": _SCHEMA}

    request_id = Column(String(255), primary_key=True)
    domain = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    feature_set_version = Column(String(50), nullable=True)
    feature_snapshot_hash = Column(String(64), nullable=True)
    entity_key = Column(String(255), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    predictions = Column(JSON, nullable=True)
    decision = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)  # Array stored as JSON
    trace_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now)


class AsyncJob(Base):
    """Model for {schema}.async_jobs"""
    __tablename__ = "async_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="status_check"),
        {"schema": _SCHEMA}
    )

    job_id = Column(String(255), primary_key=True)
    domain = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    payload = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    callback_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now, onupdate=datetime.now)
