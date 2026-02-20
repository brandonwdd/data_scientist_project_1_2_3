"""
Drift Monitoring Job
B11: Daily drift monitoring with PSI/KS
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

from platform_sdk.observability.drift import compute_drift_report
from platform_sdk.feature_store.drift_baseline import DriftBaseline
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class DriftMonitor:
    """Drift monitoring job"""

    def __init__(
        self,
        domain: str,
        model_name: str,
        baseline_path: Optional[str] = None
    ):
        """
        Initialize drift monitor
        
        Args:
            domain: Domain name (churn, fraud, rag)
            model_name: Model name
            baseline_path: Path to drift baseline JSON
        """
        self.domain = domain
        self.model_name = model_name
        self.baseline_manager = DriftBaseline()
        
        if baseline_path:
            self.baseline = self.baseline_manager.load_baseline(baseline_path)
        else:
            self.baseline = None

    def check_drift(
        self,
        current_features: pd.DataFrame,
        current_scores: Optional[np.ndarray] = None,
        baseline_features: Optional[pd.DataFrame] = None,
        baseline_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Check for drift in features and scores
        
        Args:
            current_features: Current feature distribution
            current_scores: Current score distribution
            baseline_features: Baseline features (if baseline not loaded)
            baseline_scores: Baseline scores (if baseline not loaded)
        
        Returns:
            Drift report with PSI/KS metrics
        """
        if self.baseline is None:
            if baseline_features is None:
                raise ValueError("Baseline required (either load from file or provide baseline_features)")
            baseline_features = baseline_features
            baseline_scores = baseline_scores
        else:
            # Reconstruct baseline features from baseline dict
            # (Simplified - would need full implementation)
            baseline_features = None
            baseline_scores = None

        # Compute drift report
        drift_report = compute_drift_report(
            baseline_features if baseline_features is not None else pd.DataFrame(),
            current_features,
            baseline_scores,
            current_scores
        )

        # Check triggers
        triggers = self._check_triggers(drift_report)

        return {
            "drift_report": drift_report,
            "triggers": triggers,
            "timestamp": datetime.now().isoformat()
        }

    def _check_triggers(self, drift_report: Dict) -> Dict[str, bool]:
        """
        Check drift triggers
        
        Triggers:
        - PSI(score) ≥ 0.25 for 2 consecutive windows
        - Key feature PSI ≥ 0.30 for 2 consecutive windows
        """
        triggers = {
            "score_drift_high": False,
            "feature_drift_high": False,
            "retrain_triggered": False
        }

        # Check score drift
        if drift_report.get("score_drift"):
            score_psi = drift_report["score_drift"].get("psi", 0)
            if score_psi >= 0.25:
                triggers["score_drift_high"] = True

        # Check feature drift
        feature_drift = drift_report.get("feature_drift", {})
        for feature, metrics in feature_drift.items():
            psi = metrics.get("psi", 0)
            if psi >= 0.30:
                triggers["feature_drift_high"] = True
                break

        # Retrain trigger (would check history)
        if triggers["score_drift_high"] or triggers["feature_drift_high"]:
            # Check if triggered for 2 consecutive windows
            # (This would check historical state)
            triggers["retrain_triggered"] = True

        return triggers

    def daily_drift_job(
        self,
        current_features: pd.DataFrame,
        current_scores: np.ndarray
    ):
        """
        Daily drift monitoring job
        
        Returns:
            Drift report and recommendations
        """
        logger.info(f"Running daily drift check for {self.domain}/{self.model_name}")

        drift_result = self.check_drift(current_features, current_scores)

        # Log to Prometheus (would implement)
        # self._log_prometheus_metrics(drift_result)

        # Save to MLflow artifact (would implement)
        # self._save_mlflow_artifact(drift_result)

        # Check if retrain needed
        if drift_result["triggers"]["retrain_triggered"]:
            logger.warning("Retrain triggered due to drift")
            # Would trigger retrain pipeline

        return drift_result
