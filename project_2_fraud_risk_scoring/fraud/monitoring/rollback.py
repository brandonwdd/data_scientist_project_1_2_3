"""
Rollback Mechanism for Fraud Service
Rollback runbook implementation (stricter SLO: 80ms p95, 0.1% 5xx)
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import mlflow

import sys
from pathlib import Path

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class ModelRollback:
    """
    Model Rollback Manager for Fraud Service
    
    Rollback trigger conditions (stricter):
    - p95 > 2×SLO (160ms) for 10min
    - 5xx > 1% for 5min
    """

    def __init__(
        self,
        domain: str = "fraud",
        model_name: str = "fraud_riskscore",
        slo_p95_ms: int = 80  # Stricter than churn (120ms)
    ):
        self.domain = domain
        self.model_name = model_name
        self.slo_p95_ms = slo_p95_ms
        self.rollback_threshold_p95 = slo_p95_ms * 2  # 160ms (2×SLO)
        self.rollback_threshold_5xx = 0.01  # 1%

    def check_rollback_conditions(
        self,
        current_p95_ms: float,
        current_5xx_rate: float,
        duration_minutes: int = 10
    ) -> Dict[str, Any]:
        """
        Check if rollback conditions are met
        
        Args:
            current_p95_ms: Current p95 latency in ms
            current_5xx_rate: Current 5xx error rate
            duration_minutes: Duration of violation
        
        Returns:
            Dictionary with rollback_triggered flag and details
        """
        conditions = {
            "p95_violation": current_p95_ms > self.rollback_threshold_p95,
            "p95_duration_minutes": duration_minutes,
            "p95_threshold": self.rollback_threshold_p95,
            "p95_slo": self.slo_p95_ms,
            "5xx_violation": current_5xx_rate > self.rollback_threshold_5xx,
            "5xx_duration_minutes": duration_minutes,
            "5xx_threshold": self.rollback_threshold_5xx
        }

        rollback_triggered = (
            (conditions["p95_violation"] and duration_minutes >= 10) or
            (conditions["5xx_violation"] and duration_minutes >= 5)
        )

        return {
            "rollback_triggered": rollback_triggered,
            "conditions": conditions,
            "timestamp": datetime.now().isoformat()
        }

    def rollback_to_previous_version(self) -> Optional[str]:
        """
        Rollback to previous production version
        
        Returns:
            Previous version name or None if no previous version
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get current production version
            current_versions = client.get_latest_versions(
                self.model_name,
                stages=["Production"]
            )
            
            if len(current_versions) == 0:
                logger.warning(f"No production version found for {self.model_name}")
                return None
            
            current_version = current_versions[0]
            
            # Get all production versions (sorted by creation time)
            all_versions = client.search_model_versions(
                f"name='{self.model_name}'"
            )
            production_versions = [
                v for v in all_versions
                if v.current_stage == "Production"
            ]
            
            if len(production_versions) <= 1:
                logger.warning(f"Only one production version, cannot rollback")
                return None
            
            # Find previous version (second most recent)
            production_versions.sort(key=lambda v: v.creation_timestamp, reverse=True)
            previous_version = production_versions[1]
            
            # Transition: current -> Archived, previous -> Production
            client.transition_model_version_stage(
                self.model_name,
                current_version.version,
                "Archived",
                archive_existing_versions=False
            )
            
            client.transition_model_version_stage(
                self.model_name,
                previous_version.version,
                "Production",
                archive_existing_versions=False
            )
            
            logger.info(
                f"Rolled back from version {current_version.version} "
                f"to version {previous_version.version}"
            )
            
            # Log incident artifact
            self._log_incident_artifact(current_version.version, previous_version.version)
            
            return previous_version.version
        
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return None

    def _log_incident_artifact(
        self,
        rolled_back_version: str,
        new_version: str
    ):
        """Log rollback incident as MLflow artifact"""
        try:
            incident = {
                "incident_type": "model_rollback",
                "domain": self.domain,
                "model_name": self.model_name,
                "rolled_back_version": rolled_back_version,
                "new_version": new_version,
                "timestamp": datetime.now().isoformat(),
                "reason": "SLO violation (p95 > 2×SLO or 5xx > 1%)"
            }
            
            # Create a run to log the incident
            with mlflow.start_run(run_name=f"rollback_incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_dict(incident, "rollback_incident.json")
                mlflow.set_tag("incident_type", "rollback")
                mlflow.set_tag("domain", self.domain)
            
            logger.info("Incident artifact logged to MLflow")
        
        except Exception as e:
            logger.error(f"Failed to log incident artifact: {e}")


def check_and_rollback_if_needed(
    current_p95_ms: float,
    current_5xx_rate: float,
    duration_minutes: int = 10
) -> bool:
    """
    Convenience function to check conditions and rollback if needed
    
    Returns:
        True if rollback was performed
    """
    rollback_manager = ModelRollback()
    
    check_result = rollback_manager.check_rollback_conditions(
        current_p95_ms, current_5xx_rate, duration_minutes
    )
    
    if check_result["rollback_triggered"]:
        logger.warning("Rollback conditions met, initiating rollback...")
        previous_version = rollback_manager.rollback_to_previous_version()
        return previous_version is not None
    
    return False
