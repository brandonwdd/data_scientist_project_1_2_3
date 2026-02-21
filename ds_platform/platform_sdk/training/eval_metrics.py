"""Evaluation Metrics: Common evaluation metrics for ML models"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    topk_percentiles: List[float] = [0.05, 0.10, 0.20]
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted classes (if None, uses threshold 0.5)
        topk_percentiles: Percentiles for top-K metrics
    
    Returns:
        Dictionary of metrics
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "brier": brier_score_loss(y_true, y_pred_proba),
        "log_loss": log_loss(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ece": compute_ece(y_true, y_pred_proba)
    }

    # TopK metrics
    for pct in topk_percentiles:
        k = int(len(y_true) * pct)
        topk_indices = np.argsort(y_pred_proba)[-k:]
        topk_precision = y_true.iloc[topk_indices].mean()
        baseline_rate = y_true.mean()
        lift = topk_precision / baseline_rate if baseline_rate > 0 else 0

        metrics[f"topk_precision_{int(pct*100)}pct"] = topk_precision
        metrics[f"lift_{int(pct*100)}pct"] = lift

    return metrics


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics
    
    Returns:
        Dictionary with MAE, RMSE, SMAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # SMAPE: Symmetric Mean Absolute Percentage Error
    smape = 100 * np.mean(
        np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape
    }


def compute_ece(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE)
    """
    y_true = y_true.values
    y_pred_proba = y_pred_proba.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
