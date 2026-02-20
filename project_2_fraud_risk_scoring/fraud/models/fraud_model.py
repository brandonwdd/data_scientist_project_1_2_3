"""
Fraud Risk Model
Risk control scoring model
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb


class FraudModel:
    """Fraud risk scoring model"""

    def __init__(
        self,
        model_type: str = "lightgbm",
        calibrate: bool = True,
        calibration_method: str = "isotonic"
    ):
        """
        Initialize fraud model
        
        Args:
            model_type: "lightgbm", "random_forest", or "isolation_forest"
            calibrate: Whether to calibrate probabilities
            calibration_method: "isotonic" or "platt"
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        
        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        lgb_params_override: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Train fraud model

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping
            lgb_params_override: Override default LightGBM params (e.g. from train_local.yaml)
        """
        self.feature_names = X_train.columns.tolist()

        if self.model_type == "lightgbm":
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if (X_val is not None and y_val is not None) else None

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "is_unbalance": True
            }
            if lgb_params_override:
                for k, v in lgb_params_override.items():
                    if k not in ("objective", "metric", "boosting_type", "verbose", "is_unbalance"):
                        params[k] = v

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data] if val_data else [train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(period=100)]
            )
            
        elif self.model_type == "random_forest":
            # Alternative: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "isolation_forest":
            # Unsupervised: Isolation Forest (for anomaly detection)
            # Only use fraud samples for training
            fraud_samples = X_train[y_train == 1]
            self.model = IsolationForest(
                contamination=0.01,  # Expected fraud rate
                random_state=42,
                n_estimators=100
            )
            self.model.fit(fraud_samples)
            # Note: Isolation Forest outputs -1 (anomaly) or 1 (normal)
            # Need to convert to probability
        
        # Calibration
        if self.calibrate and self.model_type != "isolation_forest":
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.calibrator = CalibratedClassifierCV(
                self.model,
                method=self.calibration_method,
                cv=3
            )
            # For LightGBM, we need to wrap it
            if self.model_type == "lightgbm":
                # LightGBM doesn't work directly with CalibratedClassifierCV
                # Use predict_proba directly and calibrate separately if needed
                pass
            else:
                self.calibrator.fit(X_train_scaled, y_train)
        
        # Compute training metrics
        train_pred = self.predict_proba(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred)
        
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.predict_proba(X_val)
            val_metrics = self._compute_metrics(y_val, val_pred)
        
        return {
            "train": train_metrics,
            "val": val_metrics
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability
        
        Returns:
            Array of probabilities (fraud probability)
        """
        if self.model_type == "lightgbm":
            proba = self.model.predict(X[self.feature_names])
            # LightGBM outputs probability directly
            return proba.reshape(-1, 1) if proba.ndim == 1 else proba
        
        elif self.model_type == "random_forest":
            if self.calibrator:
                X_scaled = self.scaler.transform(X)
                proba = self.calibrator.predict_proba(X_scaled)[:, 1]
            else:
                proba = self.model.predict_proba(X)[:, 1]
            return proba
        
        elif self.model_type == "isolation_forest":
            # Isolation Forest: -1 (anomaly/fraud) or 1 (normal)
            scores = self.model.decision_function(X)
            # Convert to probability (lower score = higher fraud probability)
            # Normalize to [0, 1]
            proba = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            return proba
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict fraud class
        
        Args:
            X: Features
            threshold: Decision threshold
        
        Returns:
            Array of predictions (0=legitimate, 1=fraud)
        """
        proba = self.predict_proba(X)
        if proba.ndim > 1:
            proba = proba[:, 0] if proba.shape[1] == 1 else proba[:, 1]
        return (proba >= threshold).astype(int)

    def _compute_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, brier_score_loss,
            precision_recall_curve, roc_curve
        )
        
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 0] if y_pred_proba.shape[1] == 1 else y_pred_proba[:, 1]
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
            "brier": brier_score_loss(y_true, y_pred_proba)
        }
        
        # Top-K precision
        top_5_pct = int(len(y_true) * 0.05)
        top_10_pct = int(len(y_true) * 0.10)
        
        top_5_indices = np.argsort(y_pred_proba)[-top_5_pct:]
        top_10_indices = np.argsort(y_pred_proba)[-top_10_pct:]
        
        metrics["topk_precision_5pct"] = y_true.iloc[top_5_indices].mean()
        metrics["topk_precision_10pct"] = y_true.iloc[top_10_indices].mean()
        
        return metrics

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "calibrator": self.calibrator,
            "model_type": self.model_type,
            "feature_names": self.feature_names
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> "FraudModel":
        """Load model from disk"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data["model_type"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.calibrator = model_data.get("calibrator")
        instance.feature_names = model_data["feature_names"]
        
        return instance
