"""
Rollback Mechanism
B11: Rollback runbook implementation
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import mlflow

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class ModelRollback:
    """
    Model Rollback Manager
    
    B11 Rollback trigger conditions:
    - p95 > 2×SLO for 10min
    - 5xx > 1% for 5min
    """

    def __init__(
        self,
        domain: str,
        model_name: str,
        slo_p95_ms: int = 120
    ):
        self.domain = domain
        self.model_name = model_name
        self.slo_p95_ms = slo_p95_ms
        self.rollback_threshold_p95 = slo_p95_ms * 2  # 2×SLO
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
            latest_versions = client.get_latest_versions(
                self.model_name,
                stages=["Production"]
            )
            
            if len(latest_versions) == 0:
                logger.warning(f"No production version found for {self.model_name}")
                return None
            
            if len(latest_versions) == 1:
                logger.warning(f"Only one production version, cannot rollback")
                return None
            
            # Get previous version (second latest)
            current_version = latest_versions[0]
            previous_version = latest_versions[1]
            
            # Transition: current -> Archived, previous -> Production
            client.transition_model_version_stage(
                self.model_name,
                current_version.version,
                "Archived"
            )
            
            client.transition_model_version_stage(
                self.model_name,
                previous_version.version,
                "Production"
            )
            
            logger.info(
                f"Rolled back {self.model_name} from {current_version.version} "
                f"to {previous_version.version}"
            )
            
            return previous_version.version
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return None

    def record_incident_artifact(
        self,
        incident_details: Dict[str, Any],
        mlflow_run_id: Optional[str] = None
    ):
        """
        Record incident artifact to MLflow
        
        B11 requirement: record incident artifact
        """
        try:
            incident_data = {
                "domain": self.domain,
                "model_name": self.model_name,
                "incident_type": "rollback",
                "timestamp": datetime.now().isoformat(),
                "details": incident_details
            }
            
            if mlflow_run_id:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_dict(incident_data, "incident_artifact.json")
            else:
                # Create new run for incident
                mlflow.set_experiment(f"{self.domain}/incidents")
                with mlflow.start_run():
                    mlflow.set_tags({
                        "domain": self.domain,
                        "model_name": self.model_name,
                        "incident_type": "rollback"
                    })
                    mlflow.log_dict(incident_data, "incident_artifact.json")
            
            logger.info("Incident artifact recorded")
        
        except Exception as e:
            logger.error(f"Failed to record incident artifact: {e}")
