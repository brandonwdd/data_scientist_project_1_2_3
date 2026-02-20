"""
Promotion Gate Framework
Platform generic promotion gate evaluator
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class PromotionGate:
    """Promotion gate evaluator"""

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize promotion gate
        
        Args:
            config_path: Path to YAML config file
            config: Config dictionary (alternative to config_path)
        """
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")

    def evaluate(
        self,
        metrics: Dict[str, float],
        gate_section: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if metrics pass promotion gate
        
        Args:
            metrics: Dictionary of metric name -> value
            gate_section: Section of config to evaluate (e.g., "churn", "ltv")
        
        Returns:
            Dictionary with "passed" (bool) and "details" (list of check results)
        """
        if gate_section:
            gate_config = self.config.get(gate_section, {})
        else:
            gate_config = self.config
        
        checks = []
        all_passed = True
        
        for key, threshold in gate_config.items():
            if key.endswith("_min"):
                metric_name = key.replace("_min", "")
                metric_value = metrics.get(metric_name)
                if metric_value is not None:
                    passed = metric_value >= threshold
                    checks.append({
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold,
                        "passed": passed
                    })
                    if not passed:
                        all_passed = False
            
            elif key.endswith("_max"):
                metric_name = key.replace("_max", "")
                metric_value = metrics.get(metric_name)
                if metric_value is not None:
                    passed = metric_value <= threshold
                    checks.append({
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold,
                        "passed": passed
                    })
                    if not passed:
                        all_passed = False
        
        return {
            "passed": all_passed,
            "details": checks
        }

    def promote_if_passed(
        self,
        metrics: Dict[str, float],
        model_uri: str,
        registered_model_name: str,
        gate_section: Optional[str] = None
    ) -> bool:
        """
        Evaluate gate and promote model if passed
        
        Returns:
            True if promoted, False otherwise
        """
        result = self.evaluate(metrics, gate_section)
        
        if result["passed"]:
            import mlflow
            mlflow.register_model(model_uri, registered_model_name)
            
            # A8 requirement: Gate results must be written to registry annotations/tags for auditability
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(registered_model_name, stages=["Production"])[0]
            client.set_model_version_tag(
                registered_model_name,
                model_version.version,
                "promotion_gate_passed",
                "true"
            )
            client.set_model_version_tag(
                registered_model_name,
                model_version.version,
                "promotion_gate_timestamp",
                datetime.now().isoformat()
            )
            
            return True
        
        return False


def load_gate(config_path: str) -> PromotionGate:
    """Load promotion gate from YAML file"""
    return PromotionGate(config_path=config_path)
