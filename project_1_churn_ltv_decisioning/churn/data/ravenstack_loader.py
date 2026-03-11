"""RavenStack local CSV loader: saas_churn_ltv tables → churn_ltv labels"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "saas_churn_ltv"


def load_ravenstack_tables(data_dir: Path) -> dict:
    """Load all RavenStack CSV tables into a dict."""
    data_dir = Path(data_dir)
    return {
        "accounts": pd.read_csv(data_dir / "ravenstack_accounts.csv"),
        "churn_events": pd.read_csv(data_dir / "ravenstack_churn_events.csv"),
        "feature_usage": pd.read_csv(data_dir / "ravenstack_feature_usage.csv"),
        "subscriptions": pd.read_csv(data_dir / "ravenstack_subscriptions.csv"),
        "support_tickets": pd.read_csv(data_dir / "ravenstack_support_tickets.csv"),
    }


def get_churn_labels(
    churn_events: pd.DataFrame,
    subscriptions: pd.DataFrame,
    as_of_times: List[datetime],
    horizon_days: int = 30,
) -> pd.DataFrame:
    """
    Compute churn labels: for each (account_id, as_of_time), churn=1 if the account
    churned within the next horizon_days. user_id = account_id.
    """
    churn_events = churn_events.copy()
    churn_events["churn_date"] = pd.to_datetime(churn_events["churn_date"])
    subscriptions = subscriptions.copy()
    subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
    subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"], errors="coerce")

    rows = []
    for as_of in as_of_times:
        as_of_ts = pd.Timestamp(as_of)
        window_end = as_of_ts + timedelta(days=horizon_days)

        # Accounts with active subscription at as_of (start <= as_of, end is null or > as_of)
        active_subs = subscriptions[
            (subscriptions["start_date"] <= as_of_ts) & (subscriptions["end_date"].isna() | (subscriptions["end_date"] > as_of_ts))
        ]
        account_ids = active_subs["account_id"].drop_duplicates().tolist()

        # Churn from churn_events: churn_date in (as_of, as_of + horizon_days]
        churned_from_events = set(
            churn_events[
                (churn_events["churn_date"] > as_of_ts) & (churn_events["churn_date"] <= window_end)]["account_id"].tolist()
        )

        # Churn from subscriptions: end_date in (as_of, as_of + horizon_days]
        ended_subs = subscriptions[
            (subscriptions["end_date"] > as_of_ts) & (subscriptions["end_date"] <= window_end)
        ]
        churned_from_subs = set(ended_subs["account_id"].tolist())

        for aid in account_ids:
            churned = 1 if (aid in churned_from_events or aid in churned_from_subs) else 0
            rows.append({"user_id": aid, "as_of_time": as_of_ts, "churn": churned})

    return pd.DataFrame(rows)


def get_ltv_labels(
    subscriptions: pd.DataFrame,
    as_of_times: List[datetime],
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Compute LTV labels: sum of revenue in (as_of, as_of + horizon_days] per account.
    user_id = account_id, ltv_90d = total revenue in window.
    Uses same account universe as churn (active sub at as_of) so inner merge works.
    """
    subscriptions = subscriptions.copy()
    subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
    subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"], errors="coerce")

    rows = []
    for as_of in as_of_times:
        as_of_ts = pd.Timestamp(as_of)
        window_end = as_of_ts + timedelta(days=horizon_days)

        # Accounts with active subscription at as_of (same universe as churn_labels)
        active_subs = subscriptions[
            (subscriptions["start_date"] <= as_of_ts) & (subscriptions["end_date"].isna() | (subscriptions["end_date"] > as_of_ts))
        ]
        account_ids = set(active_subs["account_id"].tolist())

        # Subscriptions that overlap with (as_of, window_end]
        overlapping = subscriptions[
            (subscriptions["start_date"] < window_end) & (subscriptions["end_date"].isna() | (subscriptions["end_date"] > as_of_ts))
        ]

        ltv_by_account = {aid: 0.0 for aid in account_ids}
        for _, row in overlapping.iterrows():
            aid = row["account_id"]
            if aid not in account_ids:
                continue
            start = row["start_date"]
            end = row["end_date"] if pd.notna(row["end_date"]) else window_end
            overlap_start = max(start, as_of_ts)
            overlap_end = min(end, window_end)
            if overlap_end <= overlap_start:
                continue
            days_overlap = (overlap_end - overlap_start).days
            if (row.get("billing_frequency") or "monthly") == "annual":
                rev = (row.get("arr_amount") or 0) * (days_overlap / 365)
            else:
                rev = (row.get("mrr_amount") or 0) * (days_overlap / 30)
            ltv_by_account[aid] += rev

        for aid, ltv in ltv_by_account.items():
            rows.append({"user_id": aid, "as_of_time": as_of_ts, "ltv_90d": ltv})

    return pd.DataFrame(rows)
