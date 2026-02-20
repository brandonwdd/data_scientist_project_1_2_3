"""
Evaluation Metrics for Fraud Risk Scoring
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    confusion_matrix, precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve


def compute_fraud_metrics(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    topk_percentiles: List[float] = [0.05, 0.10]
) -> Dict:
    """
    Compute fraud prediction metrics
    
    Returns:
        Dictionary with metrics including PR-AUC, AUC, ECE, Brier, TopK Precision, Recall@Precision
    """
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 0] if y_pred_proba.shape[1] == 1 else y_pred_proba[:, 1]
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    
    # ECE: Expected Calibration Error
    ece = compute_ece(y_true, y_pred_proba)
    
    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # TopK Precision (critical for fraud)
    topk_metrics = {}
    for pct in topk_percentiles:
        k = int(len(y_true) * pct)
        if k > 0:
            topk_indices = np.argsort(y_pred_proba)[-k:]
            topk_precision = y_true.iloc[topk_indices].mean()
            topk_metrics[f"topk_precision_{int(pct*100)}pct"] = topk_precision
    
    # Recall at fixed precision thresholds
    precision_thresholds = [0.90, 0.95]
    recall_at_precision = {}
    for prec_thresh in precision_thresholds:
        prec, rec, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # Find recall at target precision
        valid_indices = prec >= prec_thresh
        if valid_indices.any():
            recall_at_precision[f"recall_at_precision_{int(prec_thresh*100)}"] = rec[valid_indices].max()
        else:
            recall_at_precision[f"recall_at_precision_{int(prec_thresh*100)}"] = 0.0
    
    # FPR at fixed recall thresholds
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    recall_thresholds = [0.80, 0.90]
    fpr_at_recall = {}
    for rec_thresh in recall_thresholds:
        valid_indices = tpr >= rec_thresh
        if valid_indices.any():
            fpr_at_recall[f"fpr_at_recall_{int(rec_thresh*100)}"] = fpr[valid_indices].min()
        else:
            fpr_at_recall[f"fpr_at_recall_{int(rec_thresh*100)}"] = 1.0
    
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": logloss,
        "ece": ece,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **topk_metrics,
        **recall_at_precision,
        **fpr_at_recall
    }


def compute_policy_metrics(
    decisions_df: pd.DataFrame,
    actual_fraud: Optional[pd.Series] = None
) -> Dict:
    """
    Compute policy decision metrics
    
    Args:
        decisions_df: DataFrame with transaction_id, decision, risk_score
        actual_fraud: Optional Series with actual fraud labels
    
    Returns:
        Dictionary with FPR, FNR, approval_rate, manual_review_rate
    """
    total = len(decisions_df)
    if total == 0:
        return {}
    
    approve_count = (decisions_df["decision"] == "APPROVE").sum()
    reject_count = (decisions_df["decision"] == "REJECT").sum()
    manual_review_count = (decisions_df["decision"] == "MANUAL_REVIEW").sum()
    
    metrics = {
        "approval_rate": approve_count / total,
        "rejection_rate": reject_count / total,
        "manual_review_rate": manual_review_count / total,
        "total": total
    }
    
    # Compute FPR and FNR if actual labels available
    if actual_fraud is not None:
        # False Positive Rate: Approved transactions that were actually fraud
        approved_mask = decisions_df["decision"] == "APPROVE"
        if approved_mask.any():
            approved_fraud = actual_fraud[approved_mask].sum()
            approved_total = approved_mask.sum()
            metrics["false_positive_rate"] = approved_fraud / approved_total if approved_total > 0 else 0.0
        
        # False Negative Rate: Rejected transactions that were actually legitimate
        rejected_mask = decisions_df["decision"] == "REJECT"
        if rejected_mask.any():
            rejected_legitimate = (~actual_fraud[rejected_mask]).sum()
            rejected_total = rejected_mask.sum()
            metrics["false_negative_rate"] = rejected_legitimate / rejected_total if rejected_total > 0 else 0.0
    
    return metrics


def compute_ece(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)"""
    y_true = y_true.values
    y_pred_proba = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
    
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
    Generate evaluation artifacts (plots, tables) for fraud
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 0] if y_pred_proba.shape[1] == 1 else y_pred_proba[:, 1]
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Confusion Matrix
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
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
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
    
    # Precision-Recall Curve (detailed)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Detailed)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()
