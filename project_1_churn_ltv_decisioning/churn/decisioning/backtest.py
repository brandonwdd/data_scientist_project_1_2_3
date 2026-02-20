"""
Backtest Framework
Rolling window backtest for decisioning policy
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add ds_platform to path
platform_path = Path(__file__).parent.parent.parent.parent / "ds_platform" / "platform_sdk"
sys.path.insert(0, str(platform_path))

from platform_sdk.common.logging import setup_logging

from churn.decisioning.optimizer import DecisionOptimizer
from churn.evaluation.metrics import compute_decisioning_metrics

logger = setup_logging(__name__)


class DecisioningBacktest:
    """Backtest framework for decisioning policy"""

    def __init__(
        self,
        optimizer: DecisionOptimizer,
        stratify_by: List[str] = ["plan_tier", "channel"]
    ):
        """
        Initialize backtest
        
        Args:
            optimizer: Decision optimizer
            stratify_by: Columns to stratify by
        """
        self.optimizer = optimizer
        self.stratify_by = stratify_by

    def run_backtest(
        self,
        users_df: pd.DataFrame,
        churn_probs: np.ndarray,
        ltvs: np.ndarray,
        actual_outcomes: Optional[pd.DataFrame] = None,
        time_windows: Optional[List[Tuple[datetime, datetime]]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            users_df: DataFrame with user_id and metadata
            churn_probs: Churn probabilities
            ltvs: LTV values
            actual_outcomes: Optional DataFrame with actual churn/ltv outcomes
            time_windows: Optional list of (start, end) time windows
        
        Returns:
            Backtest results dictionary
        """
        # Run optimization
        decisions_df = self.optimizer.greedy_optimize(
            users_df, churn_probs, ltvs
        )

        # Compute metrics
        metrics = compute_decisioning_metrics(
            decisions_df,
            actual_outcomes,
            budget=self.optimizer.budget
        )

        # Stratified analysis
        stratified_metrics = {}
        for col in self.stratify_by:
            if col in users_df.columns:
                stratified_metrics[col] = self._stratify_metrics(
                    users_df, decisions_df, col, actual_outcomes
                )

        # Profit curve
        profit_curve = self._compute_profit_curve(
            users_df, churn_probs, ltvs, actual_outcomes
        )

        return {
            "decisions": decisions_df,
            "metrics": metrics,
            "stratified_metrics": stratified_metrics,
            "profit_curve": profit_curve
        }

    def _stratify_metrics(
        self,
        users_df: pd.DataFrame,
        decisions_df: pd.DataFrame,
        stratify_col: str,
        actual_outcomes: Optional[pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Compute metrics by stratum"""
        stratified = {}

        for stratum in users_df[stratify_col].unique():
            stratum_users = users_df[users_df[stratify_col] == stratum]["user_id"]
            stratum_decisions = decisions_df[
                decisions_df["user_id"].isin(stratum_users)
            ]

            if len(stratum_decisions) > 0:
                stratum_actual = None
                if actual_outcomes is not None:
                    stratum_actual = actual_outcomes[
                        actual_outcomes["user_id"].isin(stratum_users)
                    ]

                stratum_metrics = compute_decisioning_metrics(
                    stratum_decisions,
                    stratum_actual,
                    budget=self.optimizer.budget
                )

                stratified[stratum] = stratum_metrics

        return stratified

    def _compute_profit_curve(
        self,
        users_df: pd.DataFrame,
        churn_probs: np.ndarray,
        ltvs: np.ndarray,
        actual_outcomes: Optional[pd.DataFrame],
        budget_range: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute profit curve (profit vs budget)
        
        Returns:
            DataFrame with budget and profit columns
        """
        if budget_range is None:
            # Default: 10% to 100% of current budget
            budget_range = [
                self.optimizer.budget * pct
                for pct in np.linspace(0.1, 1.0, 10)
            ]

        profit_curve_data = []

        original_budget = self.optimizer.budget

        for budget in budget_range:
            # Temporarily set budget
            self.optimizer.budget = budget

            # Run optimization
            decisions_df = self.optimizer.greedy_optimize(
                users_df, churn_probs, ltvs
            )

            # Compute profit
            total_profit = decisions_df["expected_profit"].sum()
            total_cost = sum(
                self.optimizer.offers.get(action, type('obj', (object,), {'cost': 0})).cost
                for action in decisions_df["action"]
            )

            profit_curve_data.append({
                "budget": budget,
                "profit": total_profit,
                "cost": total_cost,
                "budget_utilization": total_cost / budget if budget > 0 else 0
            })

        # Restore original budget
        self.optimizer.budget = original_budget

        return pd.DataFrame(profit_curve_data)

    def rolling_window_backtest(
        self,
        data_loader,
        feature_engineer,
        start_date: datetime,
        end_date: datetime,
        window_months: int = 1,
        step_months: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Run rolling window backtest
        
        Args:
            data_loader: Data loader
            feature_engineer: Feature engineer
            start_date: Start date
            end_date: End date
            window_months: Window size in months
            step_months: Step size in months
        
        Returns:
            List of backtest results for each window
        """
        results = []

        current_start = start_date

        while current_start < end_date:
            window_end = current_start + timedelta(days=30 * window_months)

            if window_end > end_date:
                break

            logger.info(f"Backtesting window: {current_start} to {window_end}")

            # Load data for window
            # (This would integrate with actual data loading)
            # users_df, churn_probs, ltvs = load_window_data(...)

            # Run backtest
            # result = self.run_backtest(users_df, churn_probs, ltvs)
            # results.append(result)

            current_start += timedelta(days=30 * step_months)

        return results
