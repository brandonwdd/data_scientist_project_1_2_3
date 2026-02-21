-- ============================================
-- Fully separated: churn / fraud / rag three databases, each with only its own schema and tables
-- Server: postgres_ds → Databases: churn, fraud, rag (all created here; container uses POSTGRES_DB=postgres)
-- ============================================

CREATE DATABASE churn;
CREATE DATABASE fraud;
CREATE DATABASE rag;

-- ========== Database churn: only schema churn ==========
\connect churn
CREATE SCHEMA IF NOT EXISTS churn;

CREATE TABLE IF NOT EXISTS churn.online_features (
    domain VARCHAR(50) NOT NULL,
    entity_key VARCHAR(255) NOT NULL,
    feature_set_version VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    materialized_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ttl_seconds INTEGER,
    PRIMARY KEY (domain, entity_key, feature_set_version)
);
CREATE INDEX IF NOT EXISTS idx_churn_online_features_domain_time ON churn.online_features(domain, event_time DESC);

CREATE TABLE IF NOT EXISTS churn.prediction_audit (
    request_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    feature_set_version VARCHAR(50),
    feature_snapshot_hash VARCHAR(64),
    entity_key VARCHAR(255),
    latency_ms INTEGER,
    predictions JSONB,
    decision JSONB,
    warnings JSONB,
    trace_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_churn_prediction_audit_domain_time ON churn.prediction_audit(domain, created_at DESC);

CREATE TABLE IF NOT EXISTS churn.async_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    payload JSONB NOT NULL,
    result JSONB,
    error TEXT,
    callback_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT churn_async_jobs_status_check CHECK (status IN ('queued', 'running', 'succeeded', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_churn_async_jobs_domain_status ON churn.async_jobs(domain, status, created_at DESC);

CREATE OR REPLACE FUNCTION churn.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS update_async_jobs_updated_at ON churn.async_jobs;
CREATE TRIGGER update_async_jobs_updated_at
    BEFORE UPDATE ON churn.async_jobs
    FOR EACH ROW
    EXECUTE FUNCTION churn.update_updated_at_column();

-- ========== Database fraud: only schema fraud ==========
\connect fraud

CREATE SCHEMA IF NOT EXISTS fraud;

CREATE TABLE IF NOT EXISTS fraud.online_features (
    domain VARCHAR(50) NOT NULL,
    entity_key VARCHAR(255) NOT NULL,
    feature_set_version VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    materialized_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ttl_seconds INTEGER,
    PRIMARY KEY (domain, entity_key, feature_set_version)
);
CREATE INDEX IF NOT EXISTS idx_fraud_online_features_domain_time ON fraud.online_features(domain, event_time DESC);

CREATE TABLE IF NOT EXISTS fraud.prediction_audit (
    request_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    feature_set_version VARCHAR(50),
    feature_snapshot_hash VARCHAR(64),
    entity_key VARCHAR(255),
    latency_ms INTEGER,
    predictions JSONB,
    decision JSONB,
    warnings JSONB,
    trace_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fraud_prediction_audit_domain_time ON fraud.prediction_audit(domain, created_at DESC);

CREATE TABLE IF NOT EXISTS fraud.async_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    payload JSONB NOT NULL,
    result JSONB,
    error TEXT,
    callback_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fraud_async_jobs_status_check CHECK (status IN ('queued', 'running', 'succeeded', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_fraud_async_jobs_domain_status ON fraud.async_jobs(domain, status, created_at DESC);

CREATE OR REPLACE FUNCTION fraud.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS update_async_jobs_updated_at ON fraud.async_jobs;
CREATE TRIGGER update_async_jobs_updated_at
    BEFORE UPDATE ON fraud.async_jobs
    FOR EACH ROW
    EXECUTE FUNCTION fraud.update_updated_at_column();

-- ========== Database rag: only schema rag, 3 tables (no online_features, retrieval uses Qdrant) ==========
\connect rag

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.prediction_audit (
    request_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    feature_set_version VARCHAR(50),
    feature_snapshot_hash VARCHAR(64),
    entity_key VARCHAR(255),
    latency_ms INTEGER,
    predictions JSONB,
    decision JSONB,
    warnings JSONB,
    trace_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rag_prediction_audit_domain_time ON rag.prediction_audit(domain, created_at DESC);

CREATE TABLE IF NOT EXISTS rag.async_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    payload JSONB NOT NULL,
    result JSONB,
    error TEXT,
    callback_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT rag_async_jobs_status_check CHECK (status IN ('queued', 'running', 'succeeded', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_rag_async_jobs_domain_status ON rag.async_jobs(domain, status, created_at DESC);

CREATE OR REPLACE FUNCTION rag.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS update_async_jobs_updated_at ON rag.async_jobs;
CREATE TRIGGER update_async_jobs_updated_at
    BEFORE UPDATE ON rag.async_jobs
    FOR EACH ROW
    EXECUTE FUNCTION rag.update_updated_at_column();

CREATE TABLE IF NOT EXISTS rag.feedback (
    feedback_id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    reason VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rag_feedback_request_id ON rag.feedback(request_id);
