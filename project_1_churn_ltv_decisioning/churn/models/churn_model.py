"""
Churn Prediction Model
Baseline: Logistic Regression
Main: LightGBM/XGBoost
With calibration: Isotonic / Platt
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib


class ChurnModel:
    """Churn prediction model with calibration"""

    def __init__(
        self,
        model_type: str = "lightgbm",
        calibrate: bool = True,
        calibration_method: str = "isotonic"
    ):
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
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """
        Train churn model
        
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == "logistic":
            # Baseline: Logistic Regression
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
            
            if self.calibrate:
                self.calibrator = CalibratedClassifierCV(
                    self.model,
                    method=self.calibration_method,
                    cv=3
                )
                self.calibrator.fit(X_train_scaled, y_train)
        
        elif self.model_type == "lightgbm":
            # Main: LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = None
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data] if val_data else [train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            if self.calibrate:
                # Get raw predictions for calibration
                train_pred_raw = self.model.predict(X_train, raw_score=True)
                self.calibrator = CalibratedClassifierCV(
                    LogisticRegression(),
                    method=self.calibration_method,
                    cv=3
                )
                # Convert raw scores to probabilities for calibration
                train_pred_proba = 1 / (1 + np.exp(-train_pred_raw))
                self.calibrator.fit(
                    train_pred_proba.reshape(-1, 1),
                    y_train
                )
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
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
        """Predict churn probability"""
        if self.model_type == "logistic":
            X_scaled = self.scaler.transform(X)
            if self.calibrate and self.calibrator:
                proba = self.calibrator.predict_proba(X_scaled)[:, 1]
            else:
                proba = self.model.predict_proba(X_scaled)[:, 1]
        
        elif self.model_type == "lightgbm":
            proba = self.model.predict(X)
            if self.calibrate and self.calibrator:
                proba = self.calibrator.predict_proba(
                    proba.reshape(-1, 1)
                )[:, 1]
        
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict churn class"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Compute evaluation metrics"""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            brier_score_loss, log_loss
        )
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
            "brier": brier_score_loss(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba)
        }

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            "model": self.model,
            "calibrator": self.calibrator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "calibrate": self.calibrate
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "ChurnModel":
        """Load model from disk"""
        model_data = joblib.load(path)
        instance = cls(
            model_type=model_data["model_type"],
            calibrate=model_data["calibrate"]
        )
        instance.model = model_data["model"]
        instance.calibrator = model_data["calibrator"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        return instance
