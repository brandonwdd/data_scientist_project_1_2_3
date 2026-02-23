"""Promotion gate evaluation for model promotion criteria."""

from typing import Dict


def evaluate_promotion_gate(
    metrics: Dict,
    gate_config: Dict
) -> bool:
    """
    Evaluate if model passes promotion gate
    
    Returns:
        True if all criteria met, False otherwise
    """
    checks = []
    
    # Churn metrics
    if "pr_auc_min" in gate_config:
        checks.append(metrics.get("pr_auc", 0) >= gate_config["pr_auc_min"])
    
    if "auc_min" in gate_config:
        checks.append(metrics.get("auc", 0) >= gate_config["auc_min"])
    
    if "ece_max" in gate_config:
        checks.append(metrics.get("ece", 1) <= gate_config["ece_max"])
    
    if "brier_max" in gate_config:
        checks.append(metrics.get("brier", 1) <= gate_config["brier_max"])
    
    if "topk_precision_10pct_min" in gate_config:
        key = "topk_precision_10pct"
        checks.append(metrics.get(key, 0) >= gate_config["topk_precision_10pct_min"])
    
    if "lift_10pct_min" in gate_config:
        key = "lift_10pct"
        checks.append(metrics.get(key, 0) >= gate_config["lift_10pct_min"])
    
    return all(checks)
