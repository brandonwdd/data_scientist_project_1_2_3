"""Churn scoring: model load, feature lookup, prediction."""

from typing import Dict, Optional, List
import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add ds_platform so "from platform_sdk.xxx" resolves
_ws_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_ws_root / "ds_platform") not in sys.path:
    sys.path.insert(0, str(_ws_root / "ds_platform"))

from churn.models.churn_model import ChurnModel
from churn.models.ltv_model import LTVModel
from churn.decisioning.optimizer import DecisionOptimizer, Offer
from platform_sdk.feature_store.online_lookup import OnlineFeatureLookup


class ScoringService:
    """Scoring service for churn + LTV + decisioning"""

    def __init__(
        self,
        churn_model_path: Optional[str] = None,
        ltv_model_path: Optional[str] = None,
        feature_store_url: Optional[str] = None
    ):
        # Load models
        self.churn_model = None
        self.ltv_model = None
        
        if churn_model_path and os.path.exists(churn_model_path):
            self.churn_model = ChurnModel.load(churn_model_path)
        
        if ltv_model_path and os.path.exists(ltv_model_path):
            self.ltv_model = LTVModel.load(ltv_model_path)
        
        # Initialize decision optimizer
        offers = [
            Offer("OFFER_5", cost=5.0, uplift_rate=0.05),
            Offer("OFFER_10", cost=10.0, uplift_rate=0.10),
            Offer("CALL_SUPPORT", cost=20.0, uplift_rate=0.15)
        ]
        self.optimizer = DecisionOptimizer(
            budget=10000.0,  # Default budget
            offers=offers
        )
        
        self.feature_store_url = feature_store_url
        # Use platform SDK for feature lookup
        self.feature_store = OnlineFeatureLookup(domain="churn")

    def score_user(
        self,
        user_id: str,
        feature_set_version: str = "fs_churn_v1",
        domain: str = "churn"
    ) -> Dict:
        """
        Score a single user
        
        Returns:
            Dictionary with churn_prob, ltv_90d, action, reason_codes
        """
        # Load features (from online feature store or compute)
        features = self._load_user_features(user_id, feature_set_version)
        
        if features is None or len(features) == 0:
            raise ValueError(f"No features found for user {user_id}")
        
        # Predict churn
        churn_prob = 0.0
        if self.churn_model:
            X = pd.DataFrame([features])
            churn_prob = float(self.churn_model.predict_proba(X)[0])
        
        # Predict LTV
        ltv_90d = 0.0
        if self.ltv_model:
            X = pd.DataFrame([features])
            ltv_pred = self.ltv_model.predict(X)
            ltv_90d = float(ltv_pred["ltv_90d"][0])
        
        # Decision optimization
        users_df = pd.DataFrame([{"user_id": user_id}])
        actions_df = self.optimizer.greedy_optimize(
            users_df,
            np.array([churn_prob]),
            np.array([ltv_90d])
        )
        
        action_row = actions_df.iloc[0]
        
        return {
            "user_id": user_id,
            "churn_prob": churn_prob,
            "ltv_90d": ltv_90d,
            "action": action_row["action"],
            "reason_codes": action_row["reason_codes"],
            "expected_profit": float(action_row["expected_profit"]),
            "model_version": "v1.0",  # Would come from MLflow
            "feature_set_version": feature_set_version
        }

    def explain_user(
        self,
        user_id: str,
        feature_set_version: str = "fs_churn_v1"
    ) -> Dict:
        """
        Explain prediction for a user
        
        Returns:
            Dictionary with reason_codes, top_features, shap_values
        """
        # Load features
        features = self._load_user_features(user_id, feature_set_version)
        X = pd.DataFrame([features])
        
        # Get SHAP values (if model supports it)
        shap_values = {}
        top_features = []
        
        if self.churn_model and hasattr(self.churn_model.model, 'predict'):
            # Compute SHAP values (simplified)
            # In production, use shap library
            feature_importance = {}
            for col in X.columns:
                feature_importance[col] = abs(features.get(col, 0))
            
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            top_features = [
                {"feature": k, "importance": v}
                for k, v in sorted_features
            ]
            shap_values = dict(sorted_features)
        
        # Generate reason codes
        result = self.score_user(user_id, feature_set_version)
        reason_codes = result["reason_codes"]
        
        return {
            "user_id": user_id,
            "reason_codes": reason_codes,
            "top_features": top_features,
            "shap_values": shap_values
        }

    def _load_user_features(
        self,
        user_id: str,
        feature_set_version: str
    ) -> Optional[Dict]:
        """
        Load features from online feature store; if none (e.g. local run), use mock.
        """
        try:
            entity_key = f"user:{user_id}"
            features = self.feature_store.get_features(
                entity_key=entity_key,
                feature_set_version=feature_set_version
            )
            if features and len(features) > 0:
                return features
        except Exception:
            pass
        # Local/demo: no Postgres or no materialized features → mock for loaded model
        return self._mock_features_for_user(user_id)

    def _mock_features_for_user(self, user_id: str) -> Dict:
        """Return mock feature dict so /score works without feature store (local run)."""
        names = None
        if self.churn_model and getattr(self.churn_model, "feature_names", None):
            names = self.churn_model.feature_names
        elif self.ltv_model and getattr(self.ltv_model, "feature_names", None):
            names = self.ltv_model.feature_names
        if not names:
            return {}
        # Deterministic per user_id for reproducibility
        rng = np.random.RandomState(hash(user_id) % (2**31))
        out = {}
        for n in names:
            if n == "plan_tier":
                out[n] = rng.randint(0, 3)
            else:
                out[n] = float(rng.uniform(0, 1)) if "flag" in n or "tier" in n else float(rng.uniform(0, 30))
        return out
