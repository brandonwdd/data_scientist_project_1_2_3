"""
Local IEEE-CIS Fraud Data Loader (Option B)
Reads from fraud/data/ieee_fraud/ CSV files, returns one flat table for training.
Does not touch or replace the S3 DataLoader in load_data.py.
"""

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


# Default: data next to this module
DEFAULT_IEEE_DIR = Path(__file__).parent / "ieee_fraud"

# Columns to exclude from features (ids and label)
EXCLUDE_COLS = {"TransactionID", "isFraud"}

# Max fraction of NaN allowed per column (drop if exceeded)
MAX_MISSING_FRAC = 0.95


def load_ieee_local(
    data_dir: os.PathLike = None,
    fill_na: float = -999.0,
    drop_high_missing: bool = True,
    use_numeric_only: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load IEEE-CIS train data from local CSV and return (X, y).

    - Reads train_transaction.csv and train_identity.csv, left-joins on TransactionID.
    - Builds feature matrix X (numeric columns only by default; NaN filled) and labels y = isFraud.
    - S3 path is not used; this is a separate branch for local runs.

    Args:
        data_dir: Directory containing train_transaction.csv and train_identity.csv.
                  Default: fraud/data/ieee_fraud.
        fill_na: Value to fill NaN (default -999 for tree models).
        drop_high_missing: If True, drop columns with > MAX_MISSING_FRAC missing.
        use_numeric_only: If True, use only numeric columns as features (recommended for quick run).

    Returns:
        X: DataFrame of features (no TransactionID, no isFraud).
        y: Series of binary labels (0=legitimate, 1=fraud).
    """
    data_dir = Path(data_dir or DEFAULT_IEEE_DIR)
    path_txn = data_dir / "train_transaction.csv"
    path_id = data_dir / "train_identity.csv"

    if not path_txn.exists():
        raise FileNotFoundError(f"Expected {path_txn}. Download IEEE-CIS data to {data_dir}.")

    # Load transaction table
    df_txn = pd.read_csv(path_txn)
    if "isFraud" not in df_txn.columns:
        raise ValueError("train_transaction.csv must contain 'isFraud' column.")

    # Load identity and left join (many transactions have no identity row)
    if path_id.exists():
        df_id = pd.read_csv(path_id)
        df = df_txn.merge(df_id, on="TransactionID", how="left", suffixes=("", "_id"))
    else:
        df = df_txn.copy()

    y = df["isFraud"].astype(int)

    if use_numeric_only:
        # Use only numeric columns as features; fill NaN
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in EXCLUDE_COLS
        ]
    else:
        # Include categorical: label-encode object columns
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        for c in feature_cols:
            if df[c].dtype == object or df[c].dtype.name == "category":
                df[c] = pd.factorize(df[c], na_sentinel=-1)[0]

    X = df[feature_cols].copy()
    X = X.fillna(fill_na)

    if drop_high_missing:
        # Drop columns that were almost all NaN before fill (still have constant or weird dist)
        missing_before = (df[feature_cols].isna().mean() > MAX_MISSING_FRAC)
        drop_cols = missing_before[missing_before].index.tolist()
        if drop_cols:
            X = X.drop(columns=drop_cols)

    # Drop constant columns (optional, avoids div-by-zero in some metrics)
    const = X.columns[X.nunique() <= 1]
    if len(const) > 0:
        X = X.drop(columns=const)

    return X, y
