"""
Drift Monitoring Job
Daily drift monitoring with drift loop (continuous monitoring)
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

from platform_sdk.observability.drift import compute_psi, compute_ks
from platform_sdk.feature_store.drift_baseline import DriftBaseline
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class DriftMonitor:
    """Drift monitoring with continuous loop"""

    def __init__(self, baseline_path: Optional[str] = None):
        """
        Initialize drift monitor
        
        Args:
            baseline_path: Path to drift baseline JSON
        """
        self.baseline = None
        if baseline_path:
            baseline_manager = DriftBaseline()
            self.baseline = baseline_manager.load_baseline(baseline_path)

    def check_drift(
        self,
        current_features: pd.DataFrame,
        current_scores: Optional[np.ndarray] = None,
        baseline: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Check for data drift
        
        Args:
            current_features: Current feature distribution
            current_scores: Current model scores
            baseline: Baseline distribution (if None, use self.baseline)
        
        Returns:
            Drift report dictionary
        """
        baseline = baseline or self.baseline
        if baseline is None:
            raise ValueError("Baseline not provided")
        
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "feature_drift": {},
            "score_drift": None,
            "alerts": []
        }
        
        # Check feature drift (PSI)
        for feature_name in current_features.columns:
            if feature_name in baseline.get("feature_distributions", {}):
                current_data = current_features[feature_name].dropna()
                if len(current_data) > 0:
                    psi = compute_psi(
                        baseline_dist=baseline["feature_distributions"][feature_name],
                        current_dist=current_data,
                        feature_type="numeric"  # Simplified
                    )
                    drift_report["feature_drift"][feature_name] = {
                        "psi": float(psi),
                        "alert": psi >= 0.30  # Threshold for fraud
                    }
                    
                    if psi >= 0.30:
                        drift_report["alerts"].append({
                            "type": "feature_drift",
                            "feature": feature_name,
                            "psi": float(psi),
                            "severity": "high"
                        })
        
        # Check score drift
        if current_scores is not None and baseline.get("score_distribution"):
            score_psi = compute_psi(
                baseline_dist=baseline["score_distribution"],
                current_dist=current_scores,
                feature_type="numeric"
            )
            drift_report["score_drift"] = {
                "psi": float(score_psi),
                "alert": score_psi >= 0.25  # Threshold
            }
            
            if score_psi >= 0.25:
                drift_report["alerts"].append({
                    "type": "score_drift",
                    "psi": float(score_psi),
                    "severity": "high"
                })
        
        return drift_report

    def should_retrain(self, drift_report: Dict[str, Any], consecutive_days: int = 2) -> bool:
        """
        Determine if model should be retrained
        
        Args:
            drift_report: Drift report
            consecutive_days: Number of consecutive days with drift
        
        Returns:
            True if should retrain
        """
        # Check score drift (2 consecutive days)
        if drift_report.get("score_drift", {}).get("alert"):
            # In production, check historical reports
            return True
        
        # Check critical feature drift (2 consecutive days)
        critical_features = ["amount", "velocity_risk_score", "device_risk_score"]
        for feature in critical_features:
            if drift_report["feature_drift"].get(feature, {}).get("alert"):
                return True
        
        return False


def run_drift_job():
    """Run daily drift monitoring job"""
    logger.info("Starting drift monitoring job")
    
    # Load current production data (last 24 hours)
    # current_features = load_production_features(hours=24)
    # current_scores = load_production_scores(hours=24)
    
    # For demo, use placeholder
    current_features = pd.DataFrame()  # Would load from production
    current_scores = None
    
    # Load baseline
    baseline_path = "drift_baseline.json"  # Would come from MLflow or storage
    
    monitor = DriftMonitor(baseline_path=baseline_path)
    drift_report = monitor.check_drift(current_features, current_scores)
    
    # Log to MLflow
    # mlflow.log_dict(drift_report, "drift_report.json")
    
    # Check if retrain needed
    if monitor.should_retrain(drift_report):
        logger.warning("Drift detected - retrain recommended")
        # Trigger retrain pipeline
    
    logger.info("Drift monitoring job completed")
    return drift_report


if __name__ == "__main__":
    run_drift_job()
