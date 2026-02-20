"""
Offline Feature Join
Point-in-time join for training data
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class OfflineFeatureJoin:
    """Offline feature join with point-in-time semantics"""

    def __init__(self, entity_col: str = "entity_id", time_col: str = "event_time"):
        self.entity_col = entity_col
        self.time_col = time_col

    def point_in_time_join(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_cols: List[str],
        label_time_col: str = "label_time"
    ) -> pd.DataFrame:
        """
        Perform point-in-time join
        
        For each label, only include features where:
        feature.event_time < label.label_time
        
        Args:
            features_df: DataFrame with features and event_time
            labels_df: DataFrame with labels and label_time
            feature_cols: List of feature column names
            label_time_col: Time column in labels_df
        
        Returns:
            Merged DataFrame with features joined to labels
        """
        # Ensure time columns are datetime
        features_df[self.time_col] = pd.to_datetime(features_df[self.time_col])
        labels_df[label_time_col] = pd.to_datetime(labels_df[label_time_col])

        # Merge on entity
        merged = labels_df.merge(
            features_df,
            on=self.entity_col,
            how="left",
            suffixes=("_label", "_feature")
        )

        # Filter: only keep features where event_time < label_time
        merged = merged[
            merged[self.time_col] < merged[label_time_col]
        ]

        # For each entity, take the most recent feature snapshot before label_time
        merged = merged.sort_values([self.entity_col, label_time_col, self.time_col])
        merged = merged.groupby([self.entity_col, label_time_col]).tail(1)

        return merged

    def validate_no_future_leakage(
        self,
        df: pd.DataFrame,
        feature_time_col: str = "event_time",
        label_time_col: str = "label_time"
    ) -> bool:
        """
        Validate that no features leak from the future
        
        Returns:
            True if no leakage detected, raises ValueError otherwise
        """
        if feature_time_col not in df.columns or label_time_col not in df.columns:
            return True

        leakage = df[df[feature_time_col] >= df[label_time_col]]

        if len(leakage) > 0:
            raise ValueError(
                f"Data leakage detected: {len(leakage)} rows have "
                f"{feature_time_col} >= {label_time_col}"
            )

        return True
