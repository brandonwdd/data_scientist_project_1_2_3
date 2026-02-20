"""
MLflow Client Wrapper
Unified MLflow client with naming conventions
"""

import os
import mlflow
from typing import Dict, Optional, Any
from platform_sdk.common.config import Config


class MLflowClient:
    """MLflow client wrapper with platform conventions"""

    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri is None:
            tracking_uri = Config.MLFLOW_TRACKING_URI
        
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri

    def get_experiment_name(
        self,
        domain: str,
        model_type: str,
        dataset_version: str
    ) -> str:
        """
        Get experiment name following platform convention
        
        Format: {domain}/{model_type}/{dataset_version}
        """
        return f"{domain}/{model_type}/{dataset_version}"

    def start_run(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Start MLflow run with platform tags
        
        Required tags:
        - domain, dataset_version, feature_set_version, code_version, owner, run_mode, promotion_candidate
        """
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run = mlflow.start_run(experiment_id=experiment_id, **kwargs)
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        return run

    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        """
        Log model and optionally register
        
        Model names follow convention:
        - churn_churnrisk, churn_ltv90d
        - fraud_riskscore
        - rag_router, rag_reranker
        """
        mlflow.log_model(model, artifact_path)
        
        if registered_model_name:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                registered_model_name
            )
