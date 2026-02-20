"""
Point-in-Time Join for Fraud
Ensures training samples only use features available before label_time
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd


def point_in_time_join(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str],
    entity_col: str = "transaction_id",
    time_col: str = "event_time",
    label_time_col: str = "label_time"
) -> pd.DataFrame:
    """
    Perform point-in-time join to prevent data leakage
    
    For each label, only include features where:
    feature.event_time < label.label_time
    
    Args:
        features_df: DataFrame with features and event_time
        labels_df: DataFrame with labels and label_time
        feature_cols: List of feature column names
        entity_col: Entity identifier column (e.g., transaction_id)
        time_col: Time column in features_df
        label_time_col: Time column in labels_df (when label was determined)
    
    Returns:
        Merged DataFrame with features joined to labels
    """
    # Ensure time columns are datetime
    features_df[time_col] = pd.to_datetime(features_df[time_col])
    labels_df[label_time_col] = pd.to_datetime(labels_df[label_time_col])
    
    # Merge on entity
    merged = labels_df.merge(
        features_df,
        on=entity_col,
        how="left",
        suffixes=("_label", "_feature")
    )
    
    # Filter: only keep features where event_time < label_time
    merged = merged[
        merged[time_col] < merged[label_time_col]
    ]
    
    # For each entity, take the most recent feature snapshot before label_time
    # Group by entity and label_time, take latest feature snapshot
    merged = merged.sort_values([entity_col, label_time_col, time_col])
    merged = merged.groupby([entity_col, label_time_col]).tail(1)
    
    return merged


def validate_no_future_leakage(
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
        return True  # Skip validation if columns don't exist
    
    # Check for any features with event_time >= label_time
    leakage = df[df[feature_time_col] >= df[label_time_col]]
    
    if len(leakage) > 0:
        raise ValueError(
            f"Data leakage detected: {len(leakage)} rows have "
            f"{feature_time_col} >= {label_time_col}"
        )
    
    return True
