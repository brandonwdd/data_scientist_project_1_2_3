"""
Decisioning Layer: Budget-Constrained Profit Maximization
Baseline: Greedy algorithm
Optional: 0/1 Knapsack or ILP
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Offer:
    """Offer configuration"""
    offer_type: str  # NO_ACTION, OFFER_5, OFFER_10, CALL_SUPPORT
    cost: float
    uplift_rate: float  # Expected uplift in churn reduction
    min_segment: Optional[str] = None  # Optional segment constraint


class DecisionOptimizer:
    """Budget-constrained decision optimizer"""

    def __init__(
        self,
        budget: float,
        offers: List[Offer],
        guardrails: Optional[Dict] = None
    ):
        self.budget = budget
        self.offers = {offer.offer_type: offer for offer in offers}
        self.guardrails = guardrails or {}

    def compute_expected_profit(
        self,
        user_id: str,
        churn_prob: float,
        ltv: float,
        offer_type: str,
        user_segment: Optional[str] = None
    ) -> float:
        """
        Compute expected profit for user-offer combination
        
        Formula: P(churn) * uplift(offer, segment) * LTV - cost(offer)
        """
        if offer_type == "NO_ACTION":
            return 0.0
        
        offer = self.offers.get(offer_type)
        if not offer:
            return 0.0
        
        # Get uplift rate (may vary by segment)
        uplift = offer.uplift_rate
        if offer.min_segment and user_segment:
            # Segment-specific uplift (simplified)
            uplift = uplift * 1.1 if user_segment == offer.min_segment else uplift * 0.9
        
        # Expected profit
        expected_profit = churn_prob * uplift * ltv - offer.cost
        
        return expected_profit

    def greedy_optimize(
        self,
        users_df: pd.DataFrame,
        churn_probs: np.ndarray,
        ltvs: np.ndarray,
        user_segments: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Greedy optimization: select users by profit_per_cost
        
        Args:
            users_df: DataFrame with user_id
            churn_probs: Array of churn probabilities
            ltvs: Array of LTV values
            user_segments: Optional array of user segments
        
        Returns:
            DataFrame with user_id, action, reason_codes, expected_profit
        """
        results = []
        remaining_budget = self.budget
        
        # Compute profit_per_cost for all user-offer combinations
        candidates = []
        for idx, user_id in enumerate(users_df["user_id"]):
            churn_prob = churn_probs[idx]
            ltv = ltvs[idx]
            segment = user_segments[idx] if user_segments is not None else None
            
            # Skip low-risk users (guardrail)
            if self.guardrails.get("skip_low_risk", False):
                if churn_prob < self.guardrails.get("low_risk_threshold", 0.1):
                    results.append({
                        "user_id": user_id,
                        "action": "NO_ACTION",
                        "reason_codes": ["LOW_RISK"],
                        "expected_profit": 0.0
                    })
                    continue
            
            # Evaluate each offer
            for offer_type in self.offers.keys():
                if offer_type == "NO_ACTION":
                    continue
                
                profit = self.compute_expected_profit(
                    user_id, churn_prob, ltv, offer_type, segment
                )
                cost = self.offers[offer_type].cost
                
                if profit > 0 and cost > 0:
                    profit_per_cost = profit / cost
                    candidates.append({
                        "user_id": user_id,
                        "offer_type": offer_type,
                        "profit": profit,
                        "cost": cost,
                        "profit_per_cost": profit_per_cost,
                        "churn_prob": churn_prob,
                        "ltv": ltv
                    })
        
        # Sort by profit_per_cost (descending)
        candidates_df = pd.DataFrame(candidates)
        if len(candidates_df) == 0:
            # No candidates, return NO_ACTION for all
            return pd.DataFrame(results) if results else users_df.assign(
                action="NO_ACTION",
                reason_codes=[[]] * len(users_df),
                expected_profit=0.0
            )
        
        candidates_df = candidates_df.sort_values("profit_per_cost", ascending=False)
        
        # Select users until budget is exhausted
        selected_users = set()
        for _, row in candidates_df.iterrows():
            if row["user_id"] in selected_users:
                continue  # Each user gets at most one action (guardrail)
            
            if remaining_budget >= row["cost"]:
                # Generate reason codes
                reason_codes = self._generate_reason_codes(
                    row["churn_prob"], row["ltv"]
                )
                
                results.append({
                    "user_id": row["user_id"],
                    "action": row["offer_type"],
                    "reason_codes": reason_codes,
                    "expected_profit": row["profit"]
                })
                
                selected_users.add(row["user_id"])
                remaining_budget -= row["cost"]
        
        # Add NO_ACTION for remaining users
        for user_id in users_df["user_id"]:
            if user_id not in selected_users:
                results.append({
                    "user_id": user_id,
                    "action": "NO_ACTION",
                    "reason_codes": [],
                    "expected_profit": 0.0
                })
        
        return pd.DataFrame(results)

    def _generate_reason_codes(
        self,
        churn_prob: float,
        ltv: float
    ) -> List[str]:
        """Generate reason codes based on user characteristics"""
        codes = []
        
        if churn_prob > 0.7:
            codes.append("HIGH_CHURN_RISK")
        elif churn_prob > 0.5:
            codes.append("MEDIUM_CHURN_RISK")
        
        if ltv > 1000:
            codes.append("HIGH_VALUE")
        elif ltv < 100:
            codes.append("LOW_VALUE")
        
        # Add more business logic here
        # e.g., PAYMENT_FAIL, LOW_RECENCY, SUPPORT_FRICTION
        
        return codes

    def optimize_ilp(
        self,
        users_df: pd.DataFrame,
        churn_probs: np.ndarray,
        ltvs: np.ndarray
    ) -> pd.DataFrame:
        """
        Optimal solution using Integer Linear Programming
        
        Requires: pip install pulp
        """
        try:
            import pulp
        except ImportError:
            raise ImportError(
                "pulp is required for ILP optimization. "
                "Install with: pip install pulp"
            )
        
        # Create ILP problem
        prob = pulp.LpProblem("BudgetOptimization", pulp.LpMaximize)
        
        # Decision variables: x[i, j] = 1 if user i gets offer j
        user_ids = users_df["user_id"].tolist()
        offer_types = [o for o in self.offers.keys() if o != "NO_ACTION"]
        
        x = pulp.LpVariable.dicts(
            "assignment",
            [(i, j) for i in range(len(user_ids)) for j in offer_types],
            cat="Binary"
        )
        
        # Objective: maximize total profit
        prob += pulp.lpSum([
            self.compute_expected_profit(
                user_ids[i],
                churn_probs[i],
                ltvs[i],
                j
            ) * x[(i, j)]
            for i in range(len(user_ids))
            for j in offer_types
        ])
        
        # Constraint: budget
        prob += pulp.lpSum([
            self.offers[j].cost * x[(i, j)]
            for i in range(len(user_ids))
            for j in offer_types
        ]) <= self.budget
        
        # Constraint: each user gets at most one offer
        for i in range(len(user_ids)):
            prob += pulp.lpSum([x[(i, j)] for j in offer_types]) <= 1
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        results = []
        for i, user_id in enumerate(user_ids):
            assigned_offer = None
            for j in offer_types:
                if x[(i, j)].varValue == 1:
                    assigned_offer = j
                    break
            
            if assigned_offer:
                profit = self.compute_expected_profit(
                    user_id, churn_probs[i], ltvs[i], assigned_offer
                )
                reason_codes = self._generate_reason_codes(
                    churn_probs[i], ltvs[i]
                )
            else:
                assigned_offer = "NO_ACTION"
                profit = 0.0
                reason_codes = []
            
            results.append({
                "user_id": user_id,
                "action": assigned_offer,
                "reason_codes": reason_codes,
                "expected_profit": profit
            })
        
        return pd.DataFrame(results)
