"""
Async Tasks for Churn Service
Process batch scoring tasks using Celery
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add ds_platform so "from platform_sdk.xxx" resolves
_ws_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_ws_root / "ds_platform") not in sys.path:
    sys.path.insert(0, str(_ws_root / "ds_platform"))

from platform_sdk.serving.async_queue import create_async_task, job_manager
from platform_sdk.db.audit_writer import write_audit
from platform_sdk.schemas.audit import AuditRecord
from churn.serving.scoring import ScoringService
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
@create_async_task("churn.score_batch", "churn")
def score_batch_task(job_id: str, domain: str, payload: Dict[str, Any]):
    """
    Batch scoring task
    
    Payload structure:
    {
        "user_ids": List[str],
        "feature_set_version": str (optional, default "fs_churn_v1")
    }
    """
    scoring_service = get_scoring_service()
    
    user_ids = payload.get("user_ids", [])
    feature_set_version = payload.get("feature_set_version", "fs_churn_v1")
    
    logger.info(f"Processing batch job {job_id} for {len(user_ids)} users")
    
    results = []
    errors = []
    
    for user_id in user_ids:
        try:
            # Score user
            result = scoring_service.score_user(
                user_id=user_id,
                feature_set_version=feature_set_version
            )
            
            # Write audit record for each prediction
            audit_record = AuditRecord(
                request_id=f"{job_id}_{user_id}",
                domain=domain,
                model_name="churn_churnrisk",
                model_version=result.get("model_version", "unknown"),
                feature_set_version=feature_set_version,
                entity_key=user_id,
                predictions={
                    "churn_prob": result["churn_prob"],
                    "ltv_90d": result["ltv_90d"]
                },
                decision={
                    "action": result["action"],
                    "reason_codes": result["reason_codes"]
                }
            )
            write_audit(audit_record)
            
            results.append({
                "user_id": user_id,
                "churn_prob": result["churn_prob"],
                "ltv_90d": result["ltv_90d"],
                "action": result["action"],
                "reason_codes": result["reason_codes"]
            })
        
        except Exception as e:
            error_msg = f"Failed to score user {user_id}: {str(e)}"
            logger.error(error_msg)
            errors.append({
                "user_id": user_id,
                "error": error_msg
            })
    
    # Return results
    return {
        "total": len(user_ids),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }
