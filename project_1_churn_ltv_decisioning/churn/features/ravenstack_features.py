"""
RavenStack feature engineering (pandas only).
Output columns match feature_spec: active_days_30d, sessions_7d, ... plan_tier.
Used for local runs; S3/Spark path unchanged.
"""

from typing import List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

FEATURE_NAMES = [
    "active_days_30d",
    "sessions_7d",
    "core_feature_cnt_14d",
    "payment_fail_30d",
    "price_change_flag",
    "discount_used_90d",
    "tickets_30d",
    "avg_resolution_hours_90d",
    "high_sev_ticket_30d",
    "usage_slope_4w",
    "recency_days",
    "avg_revenue_90d",
    "plan_tier",
]


def _usage_by_account(
    feature_usage: pd.DataFrame,
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Join feature_usage to account_id via subscription_id."""
    sub = subscriptions[["subscription_id", "account_id"]].drop_duplicates()
    usage = feature_usage.merge(sub, on="subscription_id", how="inner")
    usage["usage_date"] = pd.to_datetime(usage["usage_date"])
    return usage


def compute_ravenstack_features(
    tables: dict,
    as_of_times: List[datetime],
) -> pd.DataFrame:
    """
    Compute feature matrix for each (user_id, as_of_time).
    user_id = account_id. Returns DataFrame with user_id, as_of_time, and FEATURE_NAMES.
    """
    accounts = tables["accounts"]
    subscriptions = tables["subscriptions"].copy()
    subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
    subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"], errors="coerce")
    feature_usage = tables["feature_usage"]
    support_tickets = tables["support_tickets"].copy()
    support_tickets["submitted_at"] = pd.to_datetime(support_tickets["submitted_at"])
    support_tickets["closed_at"] = pd.to_datetime(support_tickets["closed_at"], errors="coerce")

    usage = _usage_by_account(feature_usage, subscriptions)
    rows = []

    for as_of in as_of_times:
        as_of_ts = pd.Timestamp(as_of)
        # Accounts at risk at as_of (have active sub)
        active_subs = subscriptions[
            (subscriptions["start_date"] <= as_of_ts)
            & (subscriptions["end_date"].isna() | (subscriptions["end_date"] > as_of_ts))
        ]
        account_ids = active_subs["account_id"].drop_duplicates().tolist()

        for aid in account_ids:
            # --- Usage features (from feature_usage via subscriptions) ---
            acc_subs = active_subs[active_subs["account_id"] == aid]["subscription_id"].tolist()
            u = usage[usage["subscription_id"].isin(acc_subs) & (usage["usage_date"] <= as_of_ts)]

            w30 = as_of_ts - timedelta(days=30)
            active_days_30d = u[u["usage_date"] >= w30]["usage_date"].dt.date.nunique() if len(u) else 0

            w7 = as_of_ts - timedelta(days=7)
            sessions_7d = u[u["usage_date"] >= w7].groupby("usage_date").size().sum() if len(u) else 0

            w14 = as_of_ts - timedelta(days=14)
            core_feature_cnt_14d = u[u["usage_date"] >= w14]["feature_name"].nunique() if len(u) else 0

            # Usage slope: (last 7d count - first 7d count) / 4 over last 28d
            w28 = as_of_ts - timedelta(days=28)
            u28 = u[u["usage_date"] >= w28]
            if len(u28) >= 2:
                first_7 = u28[u28["usage_date"] < w28 + timedelta(days=7)]
                last_7 = u28[u28["usage_date"] >= as_of_ts - timedelta(days=7)]
                usage_slope_4w = (last_7.shape[0] - first_7.shape[0]) / 4.0
            else:
                usage_slope_4w = 0.0

            last_use = u["usage_date"].max() if len(u) else None
            recency_days = (as_of_ts - last_use).days if last_use is not None else 999

            # --- Subscription / payment features ---
            sub_acc = subscriptions[(subscriptions["account_id"] == aid) & (subscriptions["start_date"] <= as_of_ts)]
            sub_past_90 = sub_acc[sub_acc["start_date"] >= as_of_ts - timedelta(days=90)]
            price_change_flag = 1 if (sub_past_90["upgrade_flag"].any() or sub_past_90["downgrade_flag"].any()) else 0
            payment_fail_30d = 0  # not in RavenStack
            discount_used_90d = 0  # not in RavenStack

            # Avg revenue 90d: sum MRR/ARR prorated for last 90 days / 90
            rev_90 = 0.0
            w90 = as_of_ts - timedelta(days=90)
            for _, r in sub_acc.iterrows():
                start, end = r["start_date"], r["end_date"]
                if pd.isna(end):
                    end = as_of_ts
                overlap_start = max(start, w90)
                overlap_end = min(end, as_of_ts)
                if overlap_end > overlap_start:
                    days_overlap = (overlap_end - overlap_start).days
                    if (r.get("billing_frequency") or "monthly") == "annual":
                        rev_90 += (r.get("arr_amount") or 0) * (days_overlap / 365)
                    else:
                        rev_90 += (r.get("mrr_amount") or 0) * (days_overlap / 30)
            avg_revenue_90d = rev_90 / 90.0 if rev_90 else 0.0

            # Latest plan_tier at as_of: from subscription with latest start_date <= as_of
            latest_sub = sub_acc[sub_acc["start_date"] <= as_of_ts].sort_values("start_date", ascending=False)
            if len(latest_sub):
                pt = latest_sub.iloc[0]["plan_tier"]
                plan_tier = {"Basic": 0, "Pro": 1, "Enterprise": 2}.get(str(pt), 0)
            else:
                acc_row = accounts[accounts["account_id"] == aid]
                pt = acc_row.iloc[0]["plan_tier"] if len(acc_row) else "Basic"
                plan_tier = {"Basic": 0, "Pro": 1, "Enterprise": 2}.get(str(pt), 0)

            # --- Support features ---
            st = support_tickets[support_tickets["account_id"] == aid]
            st_before = st[st["submitted_at"] <= as_of_ts]
            w30_st = as_of_ts - timedelta(days=30)
            w90_st = as_of_ts - timedelta(days=90)
            tickets_30d = st_before[st_before["submitted_at"] >= w30_st].shape[0]
            st_90 = st_before[st_before["submitted_at"] >= w90_st]
            avg_resolution_hours_90d = st_90["resolution_time_hours"].mean() if len(st_90) and st_90["resolution_time_hours"].notna().any() else 0.0
            high_sev_ticket_30d = st_before[
                (st_before["submitted_at"] >= w30_st)
                & (st_before["priority"].isin(["high", "urgent"]))
            ].shape[0]

            rows.append({
                "user_id": aid,
                "as_of_time": as_of_ts,
                "active_days_30d": active_days_30d,
                "sessions_7d": sessions_7d,
                "core_feature_cnt_14d": core_feature_cnt_14d,
                "payment_fail_30d": payment_fail_30d,
                "price_change_flag": price_change_flag,
                "discount_used_90d": discount_used_90d,
                "tickets_30d": tickets_30d,
                "avg_resolution_hours_90d": avg_resolution_hours_90d,
                "high_sev_ticket_30d": high_sev_ticket_30d,
                "usage_slope_4w": usage_slope_4w,
                "recency_days": recency_days,
                "avg_revenue_90d": avg_revenue_90d,
                "plan_tier": plan_tier,
            })

    return pd.DataFrame(rows)
