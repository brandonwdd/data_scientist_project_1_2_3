"""
Audit Writer
Utility functions for writing to platform.prediction_audit table
"""

from typing import Optional
from platform_sdk.schemas.audit import AuditRecord
from platform_sdk.db.pg import get_db
from platform_sdk.db.models import PredictionAudit
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


def write_audit(audit_record: AuditRecord) -> bool:
    """
    Write audit record to platform.prediction_audit
    
    Args:
        audit_record: AuditRecord object
    
    Returns:
        True if successful, False otherwise
    """
    db = get_db()
    session = db.get_session()
    
    try:
        # Use placeholder for null so DB column is not empty
        NA = "n/a"
        feature_snapshot_hash = audit_record.feature_snapshot_hash if audit_record.feature_snapshot_hash else NA
        trace_id = audit_record.trace_id if audit_record.trace_id else NA
        warnings = audit_record.warnings if audit_record.warnings is not None else ["(none)"]

        audit_db = PredictionAudit(
            request_id=audit_record.request_id,
            domain=audit_record.domain,
            model_name=audit_record.model_name,
            model_version=audit_record.model_version or NA,
            feature_set_version=audit_record.feature_set_version or NA,
            feature_snapshot_hash=feature_snapshot_hash,
            entity_key=audit_record.entity_key or NA,
            latency_ms=audit_record.latency_ms,
            predictions=audit_record.predictions,
            decision=audit_record.decision,
            warnings=warnings,
            trace_id=trace_id,
            created_at=audit_record.created_at
        )
        
        session.add(audit_db)
        session.commit()
        
        logger.debug(f"Audit record written: {audit_record.request_id}")
        return True
    
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to write audit record {audit_record.request_id}: {e}")
        return False
    
    finally:
        session.close()


async def write_audit_async(audit_record: AuditRecord) -> bool:
    """
    Async write audit record (for FastAPI async endpoints)
    
    Args:
        audit_record: AuditRecord object
    
    Returns:
        True if successful, False otherwise
    """
    # For async context, we still use sync DB operations
    # In production, you might want to use async SQLAlchemy
    return write_audit(audit_record)
