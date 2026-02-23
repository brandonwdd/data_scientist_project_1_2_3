"""Offline eval protocol: rolling time split, stratified."""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class RollingEvaluationProtocol:
    """
    Rolling window evaluation protocol
    
    - train: past 9 months
    - val: next 1 month (tune)
    - test: next 1 month (report)
    - Roll forward monthly, report mean/variance over 3 runs
    """

    def __init__(
        self,
        train_months: int = 9,
        val_months: int = 1,
        test_months: int = 1,
        roll_forward_months: int = 1
    ):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.roll_forward_months = roll_forward_months

    def generate_splits(
        self,
        start_date: datetime,
        end_date: datetime,
        n_splits: int = 3
    ) -> List[Dict[str, Tuple[datetime, datetime]]]:
        """
        Generate rolling time splits
        
        Returns:
            List of dictionaries with 'train', 'val', 'test' time ranges
        """
        splits = []
        current_start = start_date
        
        for i in range(n_splits):
            # Train period
            train_start = current_start
            train_end = train_start + timedelta(days=30 * self.train_months)
            
            # Val period
            val_start = train_end
            val_end = val_start + timedelta(days=30 * self.val_months)
            
            # Test period
            test_start = val_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            if test_end > end_date:
                break  # Not enough data for more splits
            
            splits.append({
                "train": (train_start, train_end),
                "val": (val_start, val_end),
                "test": (test_start, test_end)
            })
            
            # Roll forward
            current_start += timedelta(days=30 * self.roll_forward_months)
        
        return splits

    def evaluate_split(
        self,
        split: Dict[str, Tuple[datetime, datetime]],
        model,
        data_loader,
        feature_engineer
    ) -> Dict:
        """
        Evaluate model on a single split
        
        Returns:
            Dictionary with metrics for train/val/test
        """
        # Load data for each period
        train_data = self._load_period_data(
            split["train"], data_loader, feature_engineer
        )
        val_data = self._load_period_data(
            split["val"], data_loader, feature_engineer
        )
        test_data = self._load_period_data(
            split["test"], data_loader, feature_engineer
        )
        
        # Train model
        model.train(
            train_data["X"], train_data["y"],
            val_data["X"], val_data["y"]
        )
        
        # Evaluate on test
        test_pred = model.predict_proba(test_data["X"])
        test_metrics = self._compute_metrics(test_data["y"], test_pred)
        
        return {
            "split": split,
            "test_metrics": test_metrics
        }

    def _load_period_data(
        self,
        period: Tuple[datetime, datetime],
        data_loader,
        feature_engineer
    ) -> Dict:
        """Load and engineer features for a time period"""
        # This would integrate with actual data loading
        # Simplified for now
        return {
            "X": pd.DataFrame(),  # Placeholder
            "y": pd.Series()      # Placeholder
        }

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Compute evaluation metrics"""
        from churn.evaluation.metrics import compute_churn_metrics
        return compute_churn_metrics(y_true, y_pred_proba)

    def aggregate_results(
        self,
        split_results: List[Dict]
    ) -> Dict:
        """
        Aggregate results across splits
        
        Returns:
            Dictionary with mean and std of metrics
        """
        all_metrics = []
        for result in split_results:
            all_metrics.append(result["test_metrics"])
        
        metrics_df = pd.DataFrame(all_metrics)
        
        aggregated = {}
        for col in metrics_df.columns:
            aggregated[f"{col}_mean"] = metrics_df[col].mean()
            aggregated[f"{col}_std"] = metrics_df[col].std()
        
        return aggregated

    def stratified_evaluation(
        self,
        results_df: pd.DataFrame,
        stratify_by: List[str] = ["plan_tier", "channel"]
    ) -> Dict:
        """
        Stratified evaluation by plan_tier and channel
        
        Returns:
            Dictionary with metrics per stratum
        """
        stratified_metrics = {}
        
        for col in stratify_by:
            if col not in results_df.columns:
                continue
            
            for stratum in results_df[col].unique():
                stratum_data = results_df[results_df[col] == stratum]
                # Compute metrics for this stratum
                # (implementation depends on actual data structure)
                stratified_metrics[f"{col}_{stratum}"] = {}
        
        return stratified_metrics
