"""
LTV Prediction Model
Option A: BG/NBD + Gamma-Gamma (statistical)
Option B: LightGBM regression + calibration
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib


class LTVModel:
    """LTV prediction model"""

    def __init__(
        self,
        model_type: str = "lightgbm",
        output_quantiles: bool = True
    ):
        self.model_type = model_type
        self.output_quantiles = output_quantiles
        self.model = None
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
        Train LTV model
        
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == "lightgbm":
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = None
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                "objective": "regression",
                "metric": "rmse",
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
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Compute training metrics
        train_pred = self.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred["ltv_90d"])
        
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_pred["ltv_90d"])
        
        return {
            "train": train_metrics,
            "val": val_metrics
        }

    def predict(
        self,
        X: pd.DataFrame,
        return_quantiles: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict LTV
        
        Returns:
            Dictionary with predictions and optionally quantiles
        """
        if return_quantiles is None:
            return_quantiles = self.output_quantiles
        
        if self.model_type == "lightgbm":
            pred = self.model.predict(X)
            # Ensure non-negative
            pred = np.maximum(pred, 0)
        
        result = {"ltv_90d": pred}
        
        if return_quantiles:
            # Estimate quantiles using prediction intervals
            # Simplified: use empirical distribution of residuals
            # In production, use quantile regression or prediction intervals
            result["ltv_p10"] = pred * 0.5  # Simplified
            result["ltv_p50"] = pred
            result["ltv_p90"] = pred * 1.5  # Simplified
        
        return result

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute evaluation metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # SMAPE: Symmetric Mean Absolute Percentage Error
        smape = 100 * np.mean(
            np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
        
        return {
            "mae": mae,
            "rmse": rmse,
            "smape": smape
        }

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "output_quantiles": self.output_quantiles
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "LTVModel":
        """Load model from disk"""
        model_data = joblib.load(path)
        instance = cls(
            model_type=model_data["model_type"],
            output_quantiles=model_data["output_quantiles"]
        )
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        return instance
