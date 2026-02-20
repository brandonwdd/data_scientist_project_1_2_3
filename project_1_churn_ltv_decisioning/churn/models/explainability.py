"""
Explainability Module
SHAP explanations for Churn and LTV models
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """Model explainability using SHAP"""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer
        
        Args:
            model: Trained model (LightGBM, XGBoost, etc.)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def create_explainer(
        self,
        X_background: Optional[pd.DataFrame] = None,
        explainer_type: str = "tree"
    ):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for TreeExplainer
            explainer_type: Type of explainer (tree, kernel, linear)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available")

        if explainer_type == "tree":
            # TreeExplainer for tree-based models
            if X_background is None:
                # Use model's training data if available
                X_background = pd.DataFrame(
                    np.zeros((100, len(self.feature_names))),
                    columns=self.feature_names
                )
            self.explainer = shap.TreeExplainer(self.model, X_background)
        
        elif explainer_type == "kernel":
            # KernelExplainer (slower but more general)
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                X_background
            )
        
        elif explainer_type == "linear":
            # LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)

    def explain_instance(
        self,
        X_instance: pd.DataFrame,
        return_values: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single instance
        
        Args:
            X_instance: Single row DataFrame
            return_values: Whether to return SHAP values
        
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            self.create_explainer()

        shap_values = self.explainer.shap_values(X_instance)

        # Handle binary classification (returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Get feature importance (absolute SHAP values)
        feature_importance = {
            self.feature_names[i]: float(abs(shap_values[0][i]))
            for i in range(len(self.feature_names))
        }

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top features
        top_features = [
            {"feature": name, "importance": importance}
            for name, importance in sorted_features[:10]
        ]

        result = {
            "top_features": top_features,
            "feature_importance": feature_importance
        }

        if return_values:
            result["shap_values"] = {
                self.feature_names[i]: float(shap_values[0][i])
                for i in range(len(self.feature_names))
            }

        return result

    def explain_global(
        self,
        X_sample: pd.DataFrame,
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """
        Global explanation (feature importance)
        
        Args:
            X_sample: Sample data for explanation
            max_evals: Maximum evaluations (for KernelExplainer)
        
        Returns:
            Dictionary with global feature importance
        """
        if self.explainer is None:
            self.create_explainer(X_sample)

        shap_values = self.explainer.shap_values(X_sample)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_importance = {
            self.feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(self.feature_names))
        }

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "feature_importance": feature_importance,
            "top_features": [
                {"feature": name, "importance": importance}
                for name, importance in sorted_features
            ]
        }

    def generate_reason_codes(
        self,
        X_instance: pd.DataFrame,
        threshold: float = 0.1
    ) -> List[str]:
        """
        Generate reason codes from SHAP values
        
        Args:
            X_instance: Single row DataFrame
            threshold: Minimum SHAP value to include
        
        Returns:
            List of reason codes
        """
        explanation = self.explain_instance(X_instance)
        shap_values = explanation.get("shap_values", {})

        reason_codes = []

        # Map features to reason codes
        feature_to_reason = {
            "payment_fail_30d": "PAYMENT_FAIL",
            "recency_days": "LOW_RECENCY",
            "tickets_30d": "SUPPORT_FRICTION",
            "high_sev_ticket_30d": "HIGH_SEVERITY_TICKET",
            "usage_slope_4w": "DECLINING_USAGE",
            "active_days_30d": "LOW_ACTIVITY"
        }

        # Get top contributing features
        sorted_shap = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, value in sorted_shap:
            if abs(value) >= threshold:
                reason = feature_to_reason.get(feature, feature.upper())
                if reason not in reason_codes:
                    reason_codes.append(reason)

        return reason_codes
