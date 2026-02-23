"""Train churn model; MLflow integration."""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import mlflow
from platform_sdk.training.mlflow_client import MLflowClient
from platform_sdk.training.promotion_gate import PromotionGate

from churn.data.load_data import DataLoader
from churn.features.engineering import FeatureEngineer
from churn.features.point_in_time import point_in_time_join, validate_no_future_leakage
from churn.models.churn_model import ChurnModel
from churn.evaluation.metrics import compute_churn_metrics, generate_evaluation_artifacts
from churn.training.artifacts import generate_mlflow_artifacts


def main():
    """Main training function"""
    
    # Use platform MLflow client
    mlflow_client = MLflowClient()
    
    # Load configs
    config_dir = os.path.join(os.path.dirname(__file__), "../configs")
    with open(os.path.join(config_dir, "feature_spec.yaml")) as f:
        feature_spec = yaml.safe_load(f)
    
    gate_config_path = os.path.join(config_dir, "promotion_gate.yaml")
    promotion_gate = PromotionGate(config_path=gate_config_path)
    
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer(
        spark=None,  # Would create Spark session
        feature_spec=feature_spec
    )
    
    # Load data
    as_of_time = datetime(2024, 1, 1)  # Example
    features_df = feature_engineer.compute_all_features(
        data_loader, as_of_time
    )
    labels_df = data_loader.load_labels(as_of_time)
    
    # Point-in-time join
    train_df = point_in_time_join(
        features_df, labels_df,
        feature_cols=features_df.columns.tolist()
    )
    validate_no_future_leakage(train_df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = train_df[feature_engineer.feature_names]
    y = train_df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Start MLflow run using platform client
    experiment_name = mlflow_client.get_experiment_name(
        domain="churn",
        model_type="churnrisk",
        dataset_version="v1"
    )
    
    tags = {
        "domain": "churn",
        "dataset_version": "v1",
        "feature_set_version": feature_spec["feature_set_version"],
        "code_version": "1.0.0",
        "owner": "ml-team",
        "run_mode": "training",
        "promotion_candidate": "true"
    }
    
    with mlflow_client.start_run(experiment_name, tags=tags) as run:
        
        # Train model
        model = ChurnModel(model_type="lightgbm", calibrate=True)
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        y_test_pred = model.predict_proba(X_test)
        test_metrics = compute_churn_metrics(y_test, y_test_pred)
        
        # Log metrics
        for split, metrics in [("train", train_metrics["train"]), ("val", train_metrics["val"]), ("test", test_metrics)]:
            for key, value in metrics.items():
                mlflow.log_metric(f"{split}_{key}", value)
        
        # Generate artifacts
        artifact_dir = "artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        generate_evaluation_artifacts(y_test, y_test_pred, artifact_dir)
        generate_mlflow_artifacts(model, X_test, y_test, artifact_dir)
        
        # Log artifacts
        mlflow.log_artifacts(artifact_dir)
        
        # Log model
        model_path = os.path.join(artifact_dir, "model.pkl")
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Evaluate promotion gate using platform SDK
        gate_result = promotion_gate.evaluate(test_metrics, gate_section="churn")
        
        if gate_result["passed"]:
            # Register model using platform client
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow_client.log_model(
                model,
                artifact_path="model",
                registered_model_name="churn_churnrisk"
            )
            print("✓ Model promoted to registry")
            print(f"  Gate checks: {len([c for c in gate_result['details'] if c['passed']])}/{len(gate_result['details'])} passed")
        else:
            print("✗ Promotion gate failed")
            for check in gate_result["details"]:
                if not check["passed"]:
                    print(f"  ✗ {check['metric']}: {check['value']:.4f} (threshold: {check['threshold']})")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow UI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
