"""Drift helpers: PSI, KS, ECE."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6
) -> float:
    """
    Compute Population Stability Index (PSI)
    
    For continuous features: use quantile bins
    For categorical: use top-K + OTHER
    
    Args:
        expected: Expected distribution (baseline)
        actual: Actual distribution (current)
        n_bins: Number of bins for continuous features
        epsilon: Small value to avoid division by zero
    
    Returns:
        PSI value
    """
    # Handle continuous features
    if len(np.unique(expected)) > n_bins:
        # Use quantile bins (10 bins as per A9)
        bin_edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
    else:
        # Categorical: use top-K + OTHER (as per A9)
        # Get top-K values from expected
        unique_values, counts = np.unique(expected, return_counts=True)
        top_k = min(10, len(unique_values))  # Top-K (default 10)
        top_k_values = unique_values[np.argsort(counts)[-top_k:]]
        
        # Categorize: top-K or OTHER
        expected_categorized = np.where(
            np.isin(expected, top_k_values),
            expected,
            "OTHER"
        )
        actual_categorized = np.where(
            np.isin(actual, top_k_values),
            actual,
            "OTHER"
        )
        
        # Count
        expected_counts = np.bincount(
            pd.Categorical(expected_categorized, categories=list(top_k_values) + ["OTHER"]).codes
        )
        actual_counts = np.bincount(
            pd.Categorical(actual_categorized, categories=list(top_k_values) + ["OTHER"]).codes,
            minlength=len(expected_counts)
        )
    
    # Normalize to probabilities
    expected_probs = expected_counts / (len(expected) + epsilon)
    actual_probs = actual_counts / (len(actual) + epsilon)
    
    # Add epsilon to avoid log(0)
    expected_probs = expected_probs + epsilon
    actual_probs = actual_probs + epsilon
    
    # Compute PSI
    psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
    
    return float(psi)


def compute_ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov statistic
    
    Returns:
        KS statistic (max difference in CDF)
    """
    from scipy import stats
    statistic, _ = stats.ks_2samp(expected, actual)
    return float(statistic)


def compute_drift_report(
    baseline_features: pd.DataFrame,
    current_features: pd.DataFrame,
    baseline_scores: Optional[np.ndarray] = None,
    current_scores: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive drift report
    
    Returns:
        Dictionary with PSI/KS for features and scores
    """
    report = {
        "feature_drift": {},
        "score_drift": None
    }
    
    # Feature drift
    for col in baseline_features.select_dtypes(include=[np.number]).columns:
        if col in current_features.columns:
            expected = baseline_features[col].dropna()
            actual = current_features[col].dropna()
            
            if len(expected) > 0 and len(actual) > 0:
                psi = compute_psi(expected.values, actual.values)
                ks = compute_ks(expected.values, actual.values)
                
                report["feature_drift"][col] = {
                    "psi": psi,
                    "ks": ks
                }
    
    # Score drift
    if baseline_scores is not None and current_scores is not None:
        psi = compute_psi(baseline_scores, current_scores)
        ks = compute_ks(baseline_scores, current_scores)
        
        report["score_drift"] = {
            "psi": psi,
            "ks": ks
        }
    
    return report
