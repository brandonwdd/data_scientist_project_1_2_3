"""Train churn+LTV on RavenStack local CSVs (no S3/Spark)."""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "ds_platform"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from churn.data.ravenstack_loader import (
    load_ravenstack_tables,
    get_churn_labels,
    get_ltv_labels,
    DEFAULT_DATA_DIR,
)
from churn.features.ravenstack_features import compute_ravenstack_features, FEATURE_NAMES
from churn.models.churn_model import ChurnModel
from churn.models.ltv_model import LTVModel
from churn.evaluation.metrics import compute_churn_metrics, compute_ltv_metrics


def main(
    data_dir: Path = None,
    as_of_date: str = "2024-07-01",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    save_dir: Path = None,
):
    data_dir = data_dir or DEFAULT_DATA_DIR
    save_dir = save_dir or (PROJECT_ROOT / "data" / "demo_models")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("RavenStack local training (Churn + LTV)")
    print("=" * 50)
    print(f"Data dir: {data_dir}")
    print(f"As-of date: {as_of_date}")

    # Load tables
    tables = load_ravenstack_tables(data_dir)
    as_of_times = [datetime.strptime(as_of_date, "%Y-%m-%d")]

    # Features and labels
    features_df = compute_ravenstack_features(tables, as_of_times)
    churn_labels = get_churn_labels(
        tables["churn_events"], tables["subscriptions"], as_of_times, horizon_days=30
    )
    ltv_labels = get_ltv_labels(tables["subscriptions"], as_of_times, horizon_days=90)

    # Merge: one row per (user_id, as_of_time)
    labels_df = churn_labels.merge(
        ltv_labels, on=["user_id", "as_of_time"], how="inner"
    )
    train_df = features_df.merge(
        labels_df, on=["user_id", "as_of_time"], how="inner"
    )

    if len(train_df) == 0:
        raise RuntimeError("No training rows after merge. Check as_of_date and data.")

    X = train_df[FEATURE_NAMES]
    y_churn = train_df["churn"]
    y_ltv = train_df["ltv_90d"]

    # Stratified split for churn
    ix = np.arange(len(train_df))
    i_train, i_test = train_test_split(
        ix, test_size=test_size, random_state=random_state, stratify=y_churn
    )
    i_train, i_val = train_test_split(
        i_train, test_size=val_size / (1 - test_size), random_state=random_state, stratify=y_churn.iloc[i_train]
    )

    X_train = X.iloc[i_train]
    X_val = X.iloc[i_val]
    X_test = X.iloc[i_test]
    yc_train, yc_val, yc_test = y_churn.iloc[i_train], y_churn.iloc[i_val], y_churn.iloc[i_test]
    yl_train, yl_val, yl_test = y_ltv.iloc[i_train], y_ltv.iloc[i_val], y_ltv.iloc[i_test]

    # Train Churn
    print("\n[1/2] Training Churn model...")
    churn_model = ChurnModel(model_type="lightgbm", calibrate=True)
    churn_model.train(X_train, yc_train, X_val, yc_val)
    y_test_churn_pred = churn_model.predict_proba(X_test)
    churn_metrics = compute_churn_metrics(yc_test, y_test_churn_pred)
    for k, v in churn_metrics.items():
        print(f"  test_{k}: {v:.4f}")
    churn_path = save_dir / "churn_model.pkl"
    churn_model.save(str(churn_path))
    print(f"  Saved: {churn_path}")

    # Train LTV
    print("\n[2/2] Training LTV model...")
    ltv_model = LTVModel(model_type="lightgbm")
    ltv_model.train(X_train, yl_train, X_val, yl_val)
    ltv_pred = ltv_model.predict(X_test)
    ltv_metrics = compute_ltv_metrics(yl_test, ltv_pred["ltv_90d"])
    for k, v in ltv_metrics.items():
        print(f"  test_{k}: {v:.4f}")
    ltv_path = save_dir / "ltv_model.pkl"
    ltv_model.save(str(ltv_path))
    print(f"  Saved: {ltv_path}")

    print("\nDone. Start serving with:")
    print(f"  set CHURN_MODEL_PATH={churn_path}")
    print(f"  set LTV_MODEL_PATH={ltv_path}")
    print("  uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train Churn + LTV on RavenStack (local)")
    p.add_argument("--data-dir", type=Path, default=None, help="Path to saas_churn_ltv CSV dir")
    p.add_argument("--as-of-date", default="2024-07-01", help="Single as_of date (YYYY-MM-DD)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--save-dir", type=Path, default=None)
    args = p.parse_args()
    main(
        data_dir=args.data_dir,
        as_of_date=args.as_of_date,
        test_size=args.test_size,
        save_dir=args.save_dir,
    )
