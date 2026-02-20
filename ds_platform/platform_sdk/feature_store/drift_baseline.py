"""
Drift Baseline
A9.1: Drift Baseline and Report Implementation
Generate and store drift baseline (feature/score distributions from training or validation period)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class DriftBaseline:
    """Drift baseline generator and manager"""

    def __init__(self, baseline_version: Optional[str] = None):
        """
        Initialize drift baseline
        
        Args:
            baseline_version: Version identifier for baseline
        """
        self.baseline_version = baseline_version or f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def generate_baseline(
        self,
        features_df: pd.DataFrame,
        scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate drift baseline from features and scores
        
        Args:
            features_df: DataFrame with features
            scores: Optional array of model scores
            metadata: Optional metadata (domain, model_version, etc.)
        
        Returns:
            Baseline dictionary
        """
        baseline = {
            "baseline_version": self.baseline_version,
            "generated_at": datetime.now().isoformat(),
            "feature_distributions": {},
            "score_distribution": None,
            "metadata": metadata or {}
        }

        # Feature distributions
        for col in features_df.select_dtypes(include=[np.number]).columns:
            col_data = features_df[col].dropna()
            if len(col_data) > 0:
                baseline["feature_distributions"][col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "percentiles": {
                        "p10": float(col_data.quantile(0.10)),
                        "p25": float(col_data.quantile(0.25)),
                        "p50": float(col_data.quantile(0.50)),
                        "p75": float(col_data.quantile(0.75)),
                        "p90": float(col_data.quantile(0.90))
                    },
                    "sample_size": int(len(col_data))
                }

        # Categorical features
        for col in features_df.select_dtypes(include=['object', 'category']).columns:
            col_data = features_df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts().to_dict()
                baseline["feature_distributions"][col] = {
                    "value_counts": {str(k): int(v) for k, v in value_counts.items()},
                    "sample_size": int(len(col_data))
                }

        # Score distribution
        if scores is not None and len(scores) > 0:
            scores_clean = scores[~np.isnan(scores)]
            if len(scores_clean) > 0:
                baseline["score_distribution"] = {
                    "mean": float(np.mean(scores_clean)),
                    "std": float(np.std(scores_clean)),
                    "min": float(np.min(scores_clean)),
                    "max": float(np.max(scores_clean)),
                    "percentiles": {
                        "p10": float(np.percentile(scores_clean, 10)),
                        "p25": float(np.percentile(scores_clean, 25)),
                        "p50": float(np.percentile(scores_clean, 50)),
                        "p75": float(np.percentile(scores_clean, 75)),
                        "p90": float(np.percentile(scores_clean, 90))
                    },
                    "sample_size": int(len(scores_clean))
                }

        return baseline

    def save_baseline(
        self,
        baseline: Dict[str, Any],
        output_path: str
    ):
        """Save baseline to JSON file"""
        with open(output_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Drift baseline saved to {output_path}")

    def load_baseline(self, baseline_path: str) -> Dict[str, Any]:
        """Load baseline from JSON file"""
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
        logger.info(f"Drift baseline loaded from {baseline_path}")
        return baseline

    def save_to_mlflow(
        self,
        baseline: Dict[str, Any],
        mlflow_run_id: Optional[str] = None
    ):
        """
        Save baseline as MLflow artifact
        
        A9.1 Requirement: drift_baseline.json as MLflow artifact + optional database/object storage
        """
        try:
            import mlflow
            
            if mlflow_run_id:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_dict(baseline, "drift_baseline.json")
            else:
                mlflow.log_dict(baseline, "drift_baseline.json")
            
            logger.info("Drift baseline saved to MLflow")
            
            # Optional: also save to database/object storage
            # self.save_to_database(baseline)
        
        except Exception as e:
            logger.error(f"Failed to save baseline to MLflow: {e}")

    def save_to_database(
        self,
        baseline: Dict[str, Any],
        domain: str,
        model_name: str,
        model_version: str
    ):
        """Save baseline to database (optional)"""
        # This would store baseline in a dedicated table
        # For now, just log
        logger.info(f"Baseline for {domain}/{model_name}/{model_version} would be saved to database")
