"""
Evaluation Protocol for Fraud
Rolling evaluation with time-based splits
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from fraud.evaluation.metrics import compute_fraud_metrics


class RollingEvaluationProtocol:
    """
    Rolling evaluation protocol for fraud models
    
    B8: Rolling splits
    - train: past 9 months
    - val: next 1 month (tune)
    - test: following 1 month (report)
    - Roll 3 times per month, report mean/std
    """

    def __init__(
        self,
        train_months: int = 9,
        val_months: int = 1,
        test_months: int = 1,
        n_splits: int = 3
    ):
        """
        Initialize rolling evaluation protocol
        
        Args:
            train_months: Number of months for training
            val_months: Number of months for validation
            test_months: Number of months for testing
            n_splits: Number of rolling splits
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.n_splits = n_splits

    def generate_splits(
        self,
        start_date: datetime,
        end_date: datetime,
        n_splits: Optional[int] = None
    ) -> List[Dict[str, Tuple[datetime, datetime]]]:
        """
        Generate rolling time splits
        
        Args:
            start_date: Start date for splits
            end_date: End date for splits
            n_splits: Number of splits (overrides self.n_splits)
        
        Returns:
            List of split dictionaries with train/val/test periods
        """
        n_splits = n_splits or self.n_splits
        splits = []
        
        total_months = self.train_months + self.val_months + self.test_months
        split_duration = timedelta(days=30 * total_months)
        
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
            
            # Check if test_end exceeds end_date
            if test_end > end_date:
                break
            
            splits.append({
                "split_id": i + 1,
                "train": (train_start, train_end),
                "val": (val_start, val_end),
                "test": (test_start, test_end)
            })
            
            # Move to next split (overlap by train_months)
            current_start = train_start + timedelta(days=30)  # 1 month step
        
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
        return compute_fraud_metrics(y_true, y_pred_proba)

    def aggregate_results(
        self,
        split_results: List[Dict]
    ) -> Dict:
        """
        Aggregate results across splits
        
        Returns:
            Dictionary with mean and std of metrics
        """
        if len(split_results) == 0:
            return {}
        
        # Extract all metrics
        all_metrics = [result["test_metrics"] for result in split_results]
        
        # Aggregate
        aggregated = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))
        
        return aggregated
