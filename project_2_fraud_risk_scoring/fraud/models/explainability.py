"""SHAP explainability for fraud risk scores."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class FraudExplainability:
    """Explainability for fraud risk predictions"""

    def __init__(self, model):
        """
        Initialize explainability
        
        Args:
            model: FraudModel instance
        """
        self.model = model

    def explain_transaction(
        self,
        transaction_features: pd.DataFrame,
        top_k: int = 10
    ) -> Dict:
        """
        Explain fraud risk prediction for a transaction
        
        Args:
            transaction_features: Single transaction features (1 row DataFrame)
            top_k: Number of top features to return
        
        Returns:
            Dictionary with reason_codes, top_features, shap_values
        """
        # Get prediction
        risk_score = self.model.predict_proba(transaction_features)[0]
        if isinstance(risk_score, np.ndarray):
            risk_score = float(risk_score[0]) if risk_score.ndim > 0 else float(risk_score)
        
        # Compute SHAP values (simplified - would use actual SHAP library)
        shap_values = self._compute_shap_values(transaction_features)
        
        # Sort by absolute SHAP value
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        top_features = [
            {"feature": k, "shap_value": v, "importance": abs(v)}
            for k, v in sorted_features
        ]
        
        # Generate reason codes
        reason_codes = self._generate_reason_codes(
            transaction_features.iloc[0],
            risk_score,
            shap_values
        )
        
        return {
            "risk_score": risk_score,
            "reason_codes": reason_codes,
            "top_features": top_features,
            "shap_values": dict(sorted_features)
        }

    def _compute_shap_values(
        self,
        transaction_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute SHAP values (simplified version)
        
        In production, would use:
        - shap.TreeExplainer for LightGBM
        - shap.LinearExplainer for Logistic Regression
        """
        # Simplified: use feature values as proxy for importance
        # In production, use actual SHAP library
        shap_values = {}
        
        for col in transaction_features.columns:
            value = transaction_features[col].iloc[0]
            # Simple heuristic: higher absolute value = higher importance
            # Would be replaced with actual SHAP computation
            if pd.notna(value):
                shap_values[col] = float(value) * 0.1  # Simplified
        
        return shap_values

    def _generate_reason_codes(
        self,
        features: pd.Series,
        risk_score: float,
        shap_values: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable reason codes"""
        reason_codes = []
        
        # High risk score
        if risk_score >= 0.9:
            reason_codes.append("HIGH_RISK_SCORE")
        elif risk_score >= 0.7:
            reason_codes.append("MEDIUM_RISK_SCORE")
        
        # Feature-based reasons
        if "velocity_risk_score" in features and features["velocity_risk_score"] > 0.8:
            reason_codes.append("HIGH_VELOCITY")
        
        if "device_risk_score" in features and features["device_risk_score"] > 0.7:
            reason_codes.append("SUSPICIOUS_DEVICE")
        
        if "is_new_device" in features and features.get("is_new_device", False):
            reason_codes.append("NEW_DEVICE")
        
        if "country_change_flag_24h" in features and features.get("country_change_flag_24h", False):
            reason_codes.append("COUNTRY_CHANGE")
        
        if "chargeback_count_90d" in features and features.get("chargeback_count_90d", 0) > 0:
            reason_codes.append("HISTORICAL_CHARGEBACK")
        
        # Amount-based
        if "amount_usd" in features and features.get("amount_usd", 0) > 10000:
            reason_codes.append("HIGH_AMOUNT")
        
        return reason_codes
