"""Drift reporter: Prometheus metrics + MLflow artifact."""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from platform_sdk.observability.drift import compute_drift_report
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class DriftReporter:
    """
    Drift Reporter
    
    A9.1 Requirements:
    - Prometheus metrics (online monitoring)
    - MLflow artifact (daily/window reports)
    """

    def __init__(self, domain: str, model_name: str):
        self.domain = domain
        self.model_name = model_name

    def report_drift(
        self,
        drift_report: Dict[str, Any],
        mlflow_run_id: Optional[str] = None,
        window_name: Optional[str] = None
    ):
        """
        Report drift to Prometheus and MLflow
        
        Args:
            drift_report: Drift report from compute_drift_report
            mlflow_run_id: Optional MLflow run ID
            window_name: Window identifier (e.g., "2024-01-26")
        """
        # Export to Prometheus
        self._export_prometheus_metrics(drift_report)
        
        # Save to MLflow artifact
        if mlflow_run_id:
            self._save_mlflow_artifact(drift_report, mlflow_run_id, window_name)

    def _export_prometheus_metrics(self, drift_report: Dict[str, Any]):
        """Export drift metrics to Prometheus"""
        try:
            from prometheus_client import Gauge
            
            # Score drift PSI
            if drift_report.get("score_drift"):
                score_psi = drift_report["score_drift"].get("psi", 0)
                score_ks = drift_report["score_drift"].get("ks", 0)
                
                psi_gauge = Gauge(
                    "drift_score_psi",
                    "PSI for model scores",
                    ["domain", "model_name"]
                )
                psi_gauge.labels(
                    domain=self.domain,
                    model_name=self.model_name
                ).set(score_psi)
                
                ks_gauge = Gauge(
                    "drift_score_ks",
                    "KS statistic for model scores",
                    ["domain", "model_name"]
                )
                ks_gauge.labels(
                    domain=self.domain,
                    model_name=self.model_name
                ).set(score_ks)
            
            # Feature drift PSI
            feature_drift = drift_report.get("feature_drift", {})
            for feature, metrics in feature_drift.items():
                psi = metrics.get("psi", 0)
                
                feature_psi_gauge = Gauge(
                    "drift_feature_psi",
                    "PSI for features",
                    ["domain", "model_name", "feature"]
                )
                feature_psi_gauge.labels(
                    domain=self.domain,
                    model_name=self.model_name,
                    feature=feature
                ).set(psi)
            
            logger.info("Drift metrics exported to Prometheus")
        
        except ImportError:
            logger.warning("prometheus_client not available, skipping metrics export")

    def _save_mlflow_artifact(
        self,
        drift_report: Dict[str, Any],
        mlflow_run_id: str,
        window_name: Optional[str] = None
    ):
        """Save drift report as MLflow artifact"""
        try:
            import mlflow
            
            # Create daily/window report
            report_data = {
                "domain": self.domain,
                "model_name": self.model_name,
                "window": window_name or datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "drift_report": drift_report
            }
            
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_dict(
                    report_data,
                    f"drift_report_{window_name or 'daily'}.json"
                )
            
            logger.info(f"Drift report saved to MLflow as artifact")
        
        except Exception as e:
            logger.error(f"Failed to save drift report to MLflow: {e}")
