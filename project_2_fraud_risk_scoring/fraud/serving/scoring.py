"""
Fraud Scoring Service
Handles model loading, feature lookup, and prediction
"""

from typing import Dict, Optional, Any
import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
_sdk_parent = _project_root / "ds_platform"
sys.path.insert(0, str(_sdk_parent))

from fraud.models.fraud_model import FraudModel
from fraud.policy.policy_engine import PolicyEngine

try:
    from platform_sdk.feature_store.online_lookup import OnlineFeatureLookup
except Exception:
    OnlineFeatureLookup = None


def _default_local_model_path() -> Optional[str]:
    p = _project_root / "artifacts" / "model.pkl"
    return str(p) if p.exists() else None


class ScoringService:
    """Scoring service for fraud risk + policy decision"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_store_url: Optional[str] = None
    ):
        if model_path is None and os.environ.get("USE_LOCAL_MODEL"):
            model_path = _default_local_model_path()
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = FraudModel.load(model_path)

        self.policy_engine = PolicyEngine()
        self.feature_store = None
        if OnlineFeatureLookup is not None and not os.environ.get("USE_LOCAL_MODEL"):
            try:
                self.feature_store = OnlineFeatureLookup(domain="fraud")
            except Exception:
                self.feature_store = None

    def score_transaction(
        self,
        transaction_id: str,
        feature_set_version: str = "fs_fraud_v1",
        amount_usd: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Score a single transaction.
        When features is provided (local mode), use them directly; else use feature store.
        """
        if features is not None:
            feats = dict(features)
        else:
            feats = self._load_transaction_features(transaction_id, feature_set_version) if self.feature_store else None

        if feats is None or len(feats) == 0:
            if self.model and features is not None:
                feats = features
            else:
                raise ValueError(
                    f"No features for transaction {transaction_id}. "
                    "Local mode: send 'features' in request (same keys as training)."
                )

        risk_score = 0.0
        if self.model:
            # Build DataFrame with model's feature_names; fill missing with -999
            names = getattr(self.model, "feature_names", None) or list(feats.keys())
            row = {c: feats.get(c, -999.0) for c in names}
            X = pd.DataFrame([row])
            proba = self.model.predict_proba(X)
            risk_score = float(proba.flat[0])

        amt = amount_usd if amount_usd is not None else feats.get("amount_usd", feats.get("TransactionAmt", feats.get("amount", 0.0)))
        decision_result = self.policy_engine.decide(
            risk_score=risk_score,
            features=feats,
            amount_usd=amt
        )

        return {
            "transaction_id": transaction_id,
            "risk_score": risk_score,
            "decision": decision_result["decision"],
            "reason": decision_result["reason"],
            "rule_applied": decision_result.get("rule_applied"),
            "amount_usd": amt,
            "model_version": "v1.0",
            "feature_set_version": feature_set_version,
            "metadata": decision_result.get("metadata", {})
        }

    def _load_transaction_features(
        self,
        transaction_id: str,
        feature_set_version: str
    ) -> Optional[Dict]:
        if self.feature_store is None:
            return None
        entity_key = f"transaction:{transaction_id}"
        features = self.feature_store.get_features(
            entity_key=entity_key,
            feature_set_version=feature_set_version
        )
        return features or {}
