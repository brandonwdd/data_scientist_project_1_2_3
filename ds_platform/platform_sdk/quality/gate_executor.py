"""
Data Quality Gate Executor
A8.1: Unified implementation of execution points and failure strategies
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import json

from platform_sdk.quality.ge_runner import GERunner
from platform_sdk.quality.pandera_runner import PanderaRunner
from platform_sdk.quality.contracts import DataQualityReport
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class GateExecutionPoint(Enum):
    """Data Quality Gate execution points"""
    AFTER_FEATURES = "after_features"  # After features are generated, before entering training samples
    BEFORE_PROMOTION = "before_promotion"  # Before promotion gate


class DataQualityGateExecutor:
    """
    Data Quality Gate Executor
    
    A8.1 Requirements:
    1. Execution points: after features are generated or before promotion gate
    2. Failure strategy: block training/promote + write MLflow artifact + record failure reasons + trigger notifications
    """

    def __init__(
        self,
        ge_runner: Optional[GERunner] = None,
        pandera_runner: Optional[PanderaRunner] = None
    ):
        self.ge_runner = ge_runner or GERunner()
        self.pandera_runner = pandera_runner or PanderaRunner()

    def execute_gate(
        self,
        df,
        execution_point: GateExecutionPoint,
        gate_config: Dict[str, Any],
        mlflow_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute data quality gate
        
        Args:
            df: DataFrame to validate
            execution_point: When to execute (AFTER_FEATURES or BEFORE_PROMOTION)
            gate_config: Gate configuration (missing_rate_max, schema_drift_max, etc.)
            mlflow_run_id: Optional MLflow run ID for artifact logging
        
        Returns:
            Dictionary with passed (bool), report (DataQualityReport), and action
        """
        # Run GE and Pandera checks
        ge_report = self.ge_runner.run_suite(df, suite_name="default")
        pandera_report = self.pandera_runner.validate_schema(df)
        
        # Merge reports
        combined_report = self._merge_reports(ge_report, pandera_report)
        
        # Check domain sets (if provided)
        if "domain_sets" in gate_config:
            domain_report = self.pandera_runner.validate_domain_sets(
                df, gate_config["domain_sets"]
            )
            combined_report.domain_set_violations += domain_report.domain_set_violations
            combined_report.errors.extend(domain_report.errors)
        
        # Evaluate against thresholds
        passed = self._evaluate_thresholds(combined_report, gate_config)
        
        # Save report to MLflow (even if failed)
        if mlflow_run_id:
            self._save_to_mlflow(combined_report, mlflow_run_id)
        
        # Determine action based on execution point
        action = "block" if not passed else "continue"
        
        if not passed:
            # Record failure reason to run tags
            failure_reasons = self._extract_failure_reasons(combined_report)
            if mlflow_run_id:
                self._log_failure_tags(mlflow_run_id, failure_reasons, execution_point)
            
            # Trigger notification (would implement)
            self._trigger_notification(combined_report, execution_point, failure_reasons)
        
        return {
            "passed": passed,
            "report": combined_report,
            "action": action,
            "execution_point": execution_point.value,
            "failure_reasons": self._extract_failure_reasons(combined_report) if not passed else []
        }

    def _merge_reports(
        self,
        ge_report: DataQualityReport,
        pandera_report: DataQualityReport
    ) -> DataQualityReport:
        """Merge GE and Pandera reports"""
        combined = DataQualityReport()
        
        # Merge missing rates (take max)
        all_cols = set(ge_report.missing_rate.keys()) | set(pandera_report.missing_rate.keys())
        for col in all_cols:
            combined.missing_rate[col] = max(
                ge_report.missing_rate.get(col, 0),
                pandera_report.missing_rate.get(col, 0)
            )
        
        # Merge schema drift (sum)
        combined.schema_drift = ge_report.schema_drift + pandera_report.schema_drift
        
        # Merge domain violations
        combined.domain_set_violations = (
            ge_report.domain_set_violations + pandera_report.domain_set_violations
        )
        
        # Merge errors
        combined.errors = ge_report.errors + pandera_report.errors
        
        return combined

    def _evaluate_thresholds(
        self,
        report: DataQualityReport,
        gate_config: Dict[str, Any]
    ) -> bool:
        """Evaluate report against gate thresholds"""
        # Check missing rate
        missing_rate_max = gate_config.get("missing_rate_max", 0.02)
        max_missing = max(report.missing_rate.values()) if report.missing_rate else 0
        if max_missing > missing_rate_max:
            return False
        
        # Check schema drift
        schema_drift_max = gate_config.get("schema_drift_max", 0)
        if report.schema_drift > schema_drift_max:
            return False
        
        # Check domain violations
        domain_violations_max = gate_config.get("domain_set_violations_max", 0)
        if report.domain_set_violations > domain_violations_max:
            return False
        
        return True

    def _save_to_mlflow(
        self,
        report: DataQualityReport,
        mlflow_run_id: str
    ):
        """Save data quality report to MLflow artifact"""
        try:
            import mlflow
            
            report_dict = report.to_dict()
            report_json = json.dumps(report_dict, indent=2)
            
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_text(report_json, "data_quality_report.json")
            
            logger.info("Data quality report saved to MLflow")
        except Exception as e:
            logger.error(f"Failed to save report to MLflow: {e}")

    def _extract_failure_reasons(self, report: DataQualityReport) -> List[str]:
        """Extract failure reasons from report"""
        reasons = []
        
        if report.missing_rate:
            max_missing = max(report.missing_rate.values())
            if max_missing > 0.015:  # 1.5%
                reasons.append(f"missing_rate_exceeded: {max_missing:.2%}")
        
        if report.schema_drift > 0:
            reasons.append(f"schema_drift: {report.schema_drift} violations")
        
        if report.domain_set_violations > 0:
            reasons.append(f"domain_set_violations: {report.domain_set_violations}")
        
        return reasons

    def _log_failure_tags(
        self,
        mlflow_run_id: str,
        failure_reasons: List[str],
        execution_point: GateExecutionPoint
    ):
        """Log failure reasons to MLflow run tags"""
        try:
            import mlflow
            
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.set_tag("data_quality_gate_failed", "true")
                mlflow.set_tag("data_quality_gate_execution_point", execution_point.value)
                mlflow.set_tag("data_quality_failure_reasons", "; ".join(failure_reasons))
            
            logger.info(f"Failure tags logged to MLflow run {mlflow_run_id}")
        except Exception as e:
            logger.error(f"Failed to log tags: {e}")

    def _trigger_notification(
        self,
        report: DataQualityReport,
        execution_point: GateExecutionPoint,
        failure_reasons: List[str]
    ):
        """Trigger notification (Slack/Email/Alertmanager)"""
        # This would integrate with notification system
        logger.warning(
            f"Data quality gate failed at {execution_point.value}: {failure_reasons}"
        )
        # Would send to Slack/Email/Alertmanager
