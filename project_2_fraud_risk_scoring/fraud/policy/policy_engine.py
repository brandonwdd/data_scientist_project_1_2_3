"""
Policy Engine
Policy engine: makes decisions based on risk score and business rules
Decision: APPROVE / REJECT / MANUAL_REVIEW
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    """Transaction decision"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass
class PolicyRule:
    """Policy rule configuration"""
    name: str
    condition: str  # e.g., "risk_score >= 0.8"
    action: Decision
    priority: int  # Higher priority rules evaluated first


class PolicyEngine:
    """Policy engine for fraud risk decisions"""

    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        """
        Initialize policy engine
        
        Args:
            rules: List of policy rules (if None, use default rules)
        """
        self.rules = rules or self._default_rules()

    def _default_rules(self) -> List[PolicyRule]:
        """Default policy rules"""
        return [
            # High risk: reject
            PolicyRule(
                name="high_risk_reject",
                condition="risk_score >= 0.9",
                action=Decision.REJECT,
                priority=100
            ),
            # Very high amount + medium risk: manual review
            PolicyRule(
                name="high_amount_manual_review",
                condition="risk_score >= 0.7 and amount_usd >= 10000",
                action=Decision.MANUAL_REVIEW,
                priority=90
            ),
            # New device + medium risk: manual review
            PolicyRule(
                name="new_device_manual_review",
                condition="risk_score >= 0.6 and is_new_device == True",
                action=Decision.MANUAL_REVIEW,
                priority=80
            ),
            # Country change + medium risk: manual review
            PolicyRule(
                name="country_change_manual_review",
                condition="risk_score >= 0.6 and country_change_flag_24h == True",
                action=Decision.MANUAL_REVIEW,
                priority=70
            ),
            # Medium risk: manual review
            PolicyRule(
                name="medium_risk_manual_review",
                condition="risk_score >= 0.5",
                action=Decision.MANUAL_REVIEW,
                priority=60
            ),
            # Low risk: approve
            PolicyRule(
                name="low_risk_approve",
                condition="risk_score < 0.5",
                action=Decision.APPROVE,
                priority=50
            )
        ]

    def decide(
        self,
        risk_score: float,
        features: Dict[str, Any],
        amount_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make policy decision based on risk score and features
        
        Args:
            risk_score: Fraud risk score (0-1)
            features: Feature dictionary
            amount_usd: Transaction amount in USD
        
        Returns:
            Dictionary with decision, reason, and metadata
        """
        # Sort rules by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        # Evaluate rules in priority order
        for rule in sorted_rules:
            if self._evaluate_condition(rule.condition, risk_score, features, amount_usd):
                return {
                    "decision": rule.action.value,
                    "reason": rule.name,
                    "risk_score": risk_score,
                    "rule_applied": rule.name,
                    "metadata": {
                        "amount_usd": amount_usd,
                        "features_used": list(features.keys())[:5]  # First 5 features
                    }
                }
        
        # Default: approve (should not reach here if rules are complete)
        return {
            "decision": Decision.APPROVE.value,
            "reason": "default_approve",
            "risk_score": risk_score,
            "rule_applied": None,
            "metadata": {}
        }

    def _evaluate_condition(
        self,
        condition: str,
        risk_score: float,
        features: Dict[str, Any],
        amount_usd: Optional[float]
    ) -> bool:
        """
        Evaluate policy condition
        
        Args:
            condition: Condition string (e.g., "risk_score >= 0.8")
            risk_score: Risk score
            features: Feature dictionary
            amount_usd: Transaction amount
        
        Returns:
            True if condition is met
        """
        # Simple condition evaluator
        # In production, use a proper expression evaluator
        
        # Replace variables in condition
        expr = condition.replace("risk_score", str(risk_score))
        
        if amount_usd is not None:
            expr = expr.replace("amount_usd", str(amount_usd))
        
        # Replace feature references
        for key, value in features.items():
            if isinstance(value, (int, float, bool)):
                expr = expr.replace(key, str(value))
            elif isinstance(value, str):
                expr = expr.replace(f"{key} == True", "True" if value else "False")
                expr = expr.replace(f"{key} == False", "False" if value else "True")
        
        try:
            # Evaluate expression (simple version)
            # In production, use ast.literal_eval or a proper expression parser
            return eval(expr)
        except:
            return False

    def get_decision_stats(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute policy decision statistics
        
        Args:
            decisions: List of decision dictionaries
        
        Returns:
            Statistics dictionary
        """
        total = len(decisions)
        if total == 0:
            return {}
        
        approve_count = sum(1 for d in decisions if d["decision"] == Decision.APPROVE.value)
        reject_count = sum(1 for d in decisions if d["decision"] == Decision.REJECT.value)
        manual_review_count = sum(1 for d in decisions if d["decision"] == Decision.MANUAL_REVIEW.value)
        
        return {
            "approval_rate": approve_count / total,
            "rejection_rate": reject_count / total,
            "manual_review_rate": manual_review_count / total,
            "total": total
        }
