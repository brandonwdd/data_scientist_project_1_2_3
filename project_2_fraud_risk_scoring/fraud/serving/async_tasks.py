"""
Async Tasks for Fraud Service
Process batch scoring tasks using Celery
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

from platform_sdk.serving.async_queue import create_async_task, job_manager
from platform_sdk.db.audit_writer import write_audit
from platform_sdk.schemas.audit import AuditRecord
from fraud.serving.scoring import ScoringService
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)

# Initialize scoring service (will be reused across tasks)
_scoring_service = None


def get_scoring_service() -> ScoringService:
    """Get or create scoring service instance"""
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService()
    return _scoring_service


# Create async task using platform SDK decorator
@create_async_task("fraud.score_batch", "fraud")
def score_batch_task(job_id: str, domain: str, payload: Dict[str, Any]):
    """
    Batch scoring task
    
    Payload structure:
    {
        "transaction_ids": List[str],
        "feature_set_version": str (optional, default "fs_fraud_v1")
    }
    """
    scoring_service = get_scoring_service()
    
    transaction_ids = payload.get("transaction_ids", [])
    feature_set_version = payload.get("feature_set_version", "fs_fraud_v1")
    
    logger.info(f"Processing batch job {job_id} for {len(transaction_ids)} transactions")
    
    results = []
    errors = []
    
    for transaction_id in transaction_ids:
        try:
            # Score transaction
            result = scoring_service.score_transaction(
                transaction_id=transaction_id,
                feature_set_version=feature_set_version
            )
            
            # Write audit record for each prediction
            audit_record = AuditRecord(
                request_id=f"{job_id}_{transaction_id}",
                domain=domain,
                model_name="fraud_riskscore",
                model_version=result.get("model_version", "unknown"),
                feature_set_version=feature_set_version,
                entity_key=transaction_id,
                predictions={"risk_score": result["risk_score"]},
                decision={
                    "decision": result["decision"],
                    "reason": result["reason"],
                    "rule_applied": result.get("rule_applied")
                }
            )
            write_audit(audit_record)
            
            results.append({
                "transaction_id": transaction_id,
                "risk_score": result["risk_score"],
                "decision": result["decision"],
                "reason": result["reason"]
            })
        
        except Exception as e:
            error_msg = f"Failed to score transaction {transaction_id}: {str(e)}"
            logger.error(error_msg)
            errors.append({
                "transaction_id": transaction_id,
                "error": error_msg
            })
    
    # Return results
    return {
        "total": len(transaction_ids),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }
