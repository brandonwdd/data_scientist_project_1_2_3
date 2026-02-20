"""
Evaluation Metrics for Churn + LTV + Decisioning
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve


def compute_churn_metrics(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    topk_percentiles: List[float] = [0.05, 0.10, 0.20]
) -> Dict:
    """
    Compute churn prediction metrics
    
    Returns:
        Dictionary with metrics including PR-AUC, AUC, ECE, Brier, TopK Precision, Lift
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    
    # ECE: Expected Calibration Error
    ece = compute_ece(y_true, y_pred_proba)
    
    # TopK Precision and Lift
    topk_metrics = {}
    for pct in topk_percentiles:
        k = int(len(y_true) * pct)
        topk_indices = np.argsort(y_pred_proba)[-k:]
        topk_precision = y_true.iloc[topk_indices].mean()
        baseline_rate = y_true.mean()
        lift = topk_precision / baseline_rate if baseline_rate > 0 else 0
        
        topk_metrics[f"topk_precision_{int(pct*100)}pct"] = topk_precision
        topk_metrics[f"lift_{int(pct*100)}pct"] = lift
    
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": logloss,
        "ece": ece,
        **topk_metrics
    }


def compute_ltv_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict:
    """
    Compute LTV prediction metrics
    
    Returns:
        Dictionary with SMAPE, MAE, RMSE
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


def compute_decisioning_metrics(
    decisions_df: pd.DataFrame,
    actual_outcomes: Optional[pd.DataFrame] = None,
    budget: float = 10000.0
) -> Dict:
    """
    Compute decisioning metrics
    
    Args:
        decisions_df: DataFrame with user_id, action, expected_profit
        actual_outcomes: Optional DataFrame with actual churn/ltv outcomes
        budget: Total budget used
    
    Returns:
        Dictionary with profit_uplift, budget_utilization, offer_fp_rate, etc.
    """
    total_expected_profit = decisions_df["expected_profit"].sum()
    total_cost = decisions_df["cost"].sum() if "cost" in decisions_df.columns else 0.0
    
    budget_utilization = total_cost / budget if budget > 0 else 0.0
    
    # Offer mix
    offer_counts = decisions_df["action"].value_counts().to_dict()
    offer_mix = {
        f"offer_{k}_pct": v / len(decisions_df)
        for k, v in offer_counts.items()
    }
    
    # Offer FP rate (simplified: users who got offer but didn't churn)
    offer_fp_rate = None
    if actual_outcomes is not None:
        offered_users = decisions_df[
            decisions_df["action"] != "NO_ACTION"
        ]["user_id"]
        if len(offered_users) > 0:
            actual_churn = actual_outcomes[
                actual_outcomes["user_id"].isin(offered_users)
            ]["churn"].sum()
            offer_fp_rate = 1 - (actual_churn / len(offered_users))
    
    # Profit uplift (vs baseline policy)
    # Baseline: no action for anyone
    profit_uplift = total_expected_profit  # Simplified
    
    return {
        "total_expected_profit": total_expected_profit,
        "budget_utilization": budget_utilization,
        "profit_uplift": profit_uplift,
        "offer_fp_rate": offer_fp_rate,
        **offer_mix
    }


def compute_ece(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
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
    
    return ece


def generate_evaluation_artifacts(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str
):
    """
    Generate evaluation artifacts (plots, tables)
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()
    
    # Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
    plt.close()
