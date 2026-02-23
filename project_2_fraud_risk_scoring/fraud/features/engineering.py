"""Fraud feature engineering (feature_spec.yaml)."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, avg, max, min, sum,
    when, datediff, window, lag, lead
)

from fraud.data.load_data import DataLoader
from fraud.data.contracts import (
    validate_currency, validate_payment_method,
    validate_kyc_status, validate_account_status
)


class FeatureEngineer:
    """Feature engineering using Spark (or pandas fallback)"""

    def __init__(self, spark: Optional[SparkSession], feature_spec: Dict):
        self.spark = spark
        self.feature_spec = feature_spec
        self.entity = feature_spec.get("entity", "transaction_id")
        self.feature_set_version = feature_spec.get("feature_set_version", "fs_fraud_v1")
        self.feature_names = []

    def compute_transaction_features(
        self,
        transaction_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute transaction-level features
        
        Features: amount, amount_usd, currency, payment_method, merchant_category
        """
        features = transaction_df[[self.entity, "amount", "amount_usd", "currency", 
                                   "payment_method", "merchant_category"]].copy()
        
        # Rename to match feature spec
        features = features.rename(columns={
            "amount": "amount",
            "amount_usd": "amount_usd",
            "currency": "currency",
            "payment_method": "payment_method",
            "merchant_category": "merchant_category"
        })
        
        self.feature_names.extend(["amount", "amount_usd", "currency", 
                                   "payment_method", "merchant_category"])
        
        return features

    def compute_behavior_features(
        self,
        behavior_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute behavior features (transaction history)
        
        Features: transaction_count_1h/24h/7d, total_amount_24h, avg_amount_7d, max_amount_30d
        """
        if len(behavior_df) == 0:
            return pd.DataFrame({self.entity: []})
        
        behavior_df["event_time"] = pd.to_datetime(behavior_df["event_time"])
        
        # Group by user_id (or transaction_id for aggregation)
        entity_col = "user_id" if "user_id" in behavior_df.columns else self.entity
        
        features_list = []
        
        for entity_id in behavior_df[entity_col].unique():
            entity_data = behavior_df[behavior_df[entity_col] == entity_id]
            
            # Transaction count 1h
            one_hour_ago = as_of_time - timedelta(hours=1)
            count_1h = len(entity_data[entity_data["event_time"] >= one_hour_ago])
            
            # Transaction count 24h
            one_day_ago = as_of_time - timedelta(days=1)
            count_24h = len(entity_data[entity_data["event_time"] >= one_day_ago])
            
            # Transaction count 7d
            seven_days_ago = as_of_time - timedelta(days=7)
            count_7d = len(entity_data[entity_data["event_time"] >= seven_days_ago])
            
            # Total amount 24h
            total_24h = entity_data[entity_data["event_time"] >= one_day_ago]["amount"].sum()
            
            # Average amount 7d
            avg_7d = entity_data[entity_data["event_time"] >= seven_days_ago]["amount"].mean()
            
            # Max amount 30d
            thirty_days_ago = as_of_time - timedelta(days=30)
            max_30d = entity_data[entity_data["event_time"] >= thirty_days_ago]["amount"].max()
            
            features_list.append({
                entity_col: entity_id,
                "transaction_count_1h": count_1h,
                "transaction_count_24h": count_24h,
                "transaction_count_7d": count_7d,
                "total_amount_24h": total_24h if not pd.isna(total_24h) else 0.0,
                "avg_amount_7d": avg_7d if not pd.isna(avg_7d) else 0.0,
                "max_amount_30d": max_30d if not pd.isna(max_30d) else 0.0
            })
        
        features = pd.DataFrame(features_list)
        self.feature_names.extend([
            "transaction_count_1h", "transaction_count_24h", "transaction_count_7d",
            "total_amount_24h", "avg_amount_7d", "max_amount_30d"
        ])
        
        return features

    def compute_device_location_features(
        self,
        transaction_df: pd.DataFrame,
        behavior_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute device and location features
        
        Features: device_id, ip_address_country, is_new_device, country_change_flag_24h
        """
        if len(transaction_df) == 0:
            return pd.DataFrame({self.entity: []})
        
        features = transaction_df[[self.entity, "device_id", "ip_address_country"]].copy()
        
        # is_new_device: first seen in last 90 days
        if "device_id" in behavior_df.columns:
            ninety_days_ago = as_of_time - timedelta(days=90)
            device_history = behavior_df[
                (behavior_df["event_time"] >= ninety_days_ago) &
                (behavior_df["event_time"] < as_of_time)
            ]
            
            if len(device_history) > 0:
                first_seen_devices = device_history.groupby("device_id")["event_time"].min()
                features["is_new_device"] = features["device_id"].map(
                    lambda x: first_seen_devices.get(x, as_of_time) >= (as_of_time - timedelta(days=90))
                ).fillna(False)
            else:
                features["is_new_device"] = False
        else:
            features["is_new_device"] = False
        
        # country_change_flag_24h: country changed in last 24h
        if "ip_address_country" in behavior_df.columns:
            one_day_ago = as_of_time - timedelta(days=1)
            recent_behavior = behavior_df[behavior_df["event_time"] >= one_day_ago]
            
            if len(recent_behavior) > 0 and "user_id" in recent_behavior.columns:
                user_countries = recent_behavior.groupby("user_id")["ip_address_country"].nunique()
                features["country_change_flag_24h"] = features.get("user_id", pd.Series()).map(
                    lambda x: user_countries.get(x, 0) > 1
                ).fillna(False)
            else:
                features["country_change_flag_24h"] = False
        else:
            features["country_change_flag_24h"] = False
        
        self.feature_names.extend([
            "device_id", "ip_address_country", "is_new_device", "country_change_flag_24h"
        ])
        
        return features

    def compute_risk_features(
        self,
        risk_signal_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute risk signal features
        
        Features: chargeback_count_90d, fraud_flag_count_90d, velocity_risk_score, device_risk_score
        """
        if len(risk_signal_df) == 0:
            return pd.DataFrame()
        
        risk_signal_df["event_time"] = pd.to_datetime(risk_signal_df["event_time"])
        ninety_days_ago = as_of_time - timedelta(days=90)
        
        recent_signals = risk_signal_df[risk_signal_df["event_time"] >= ninety_days_ago]
        
        entity_col = "user_id" if "user_id" in risk_signal_df.columns else "transaction_id"
        
        # Chargeback count
        chargebacks = recent_signals[recent_signals["signal_type"] == "chargeback"]
        chargeback_count = chargebacks.groupby(entity_col).size().reset_index(name="chargeback_count_90d")
        
        # Fraud flag count
        fraud_flags = recent_signals[recent_signals["signal_type"] == "fraud_flag"]
        fraud_flag_count = fraud_flags.groupby(entity_col).size().reset_index(name="fraud_flag_count_90d")
        
        # Velocity risk score (simplified: based on transaction frequency)
        velocity_signals = recent_signals[recent_signals["signal_type"] == "velocity_alert"]
        velocity_risk = velocity_signals.groupby(entity_col)["signal_value"].max().reset_index(name="velocity_risk_score")
        
        # Device risk score
        device_signals = recent_signals[recent_signals["signal_type"] == "device_alert"]
        device_risk = device_signals.groupby(entity_col)["signal_value"].max().reset_index(name="device_risk_score")
        
        # Merge all risk features
        risk_features = chargeback_count.merge(
            fraud_flag_count, on=entity_col, how="outer"
        ).merge(
            velocity_risk, on=entity_col, how="outer"
        ).merge(
            device_risk, on=entity_col, how="outer"
        )
        
        risk_features = risk_features.fillna(0)
        
        self.feature_names.extend([
            "chargeback_count_90d", "fraud_flag_count_90d",
            "velocity_risk_score", "device_risk_score"
        ])
        
        return risk_features

    def compute_profile_features(
        self,
        profile_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute user profile features
        
        Features: account_age_days, kyc_status, account_status
        """
        if len(profile_df) == 0:
            return pd.DataFrame()
        
        features = profile_df[["user_id", "account_age_days", "kyc_status", "account_status"]].copy()
        
        self.feature_names.extend(["account_age_days", "kyc_status", "account_status"])
        
        return features

    def compute_all_features(
        self,
        data_loader: DataLoader,
        as_of_time: datetime,
        transaction_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for given as_of_time
        
        Returns:
            DataFrame with transaction_id and all features
        """
        # Load raw data
        transaction_df = data_loader.load_transaction_data(
            as_of_time - timedelta(days=1),
            as_of_time,
            transaction_ids
        )
        
        if len(transaction_df) == 0:
            return pd.DataFrame()
        
        behavior_df = data_loader.load_behavior_data(
            as_of_time - timedelta(days=90),
            as_of_time
        )
        
        risk_signal_df = data_loader.load_risk_signal_data(
            as_of_time - timedelta(days=90),
            as_of_time
        )
        
        # Get user_ids from transactions
        user_ids = transaction_df["user_id"].unique().tolist() if "user_id" in transaction_df.columns else []
        
        profile_df = data_loader.load_user_profile_data(
            as_of_time,
            user_ids if user_ids else None
        )
        
        # Compute feature groups
        transaction_features = self.compute_transaction_features(transaction_df, as_of_time)
        behavior_features = self.compute_behavior_features(behavior_df, as_of_time)
        device_features = self.compute_device_location_features(transaction_df, behavior_df, as_of_time)
        risk_features = self.compute_risk_features(risk_signal_df, as_of_time)
        profile_features = self.compute_profile_features(profile_df)
        
        # Join all features
        result = transaction_features.copy()
        
        # Merge behavior features (on user_id)
        if len(behavior_features) > 0 and "user_id" in transaction_df.columns:
            result = result.merge(
                behavior_features,
                left_on="user_id",
                right_on="user_id" if "user_id" in behavior_features.columns else self.entity,
                how="left"
            )
        
        # Merge device features
        if len(device_features) > 0:
            result = result.merge(
                device_features,
                on=self.entity,
                how="left"
            )
        
        # Merge risk features (on user_id)
        if len(risk_features) > 0 and "user_id" in transaction_df.columns:
            result = result.merge(
                risk_features,
                left_on="user_id",
                right_on="user_id" if "user_id" in risk_features.columns else self.entity,
                how="left"
            )
        
        # Merge profile features (on user_id)
        if len(profile_features) > 0 and "user_id" in transaction_df.columns:
            result = result.merge(
                profile_features,
                on="user_id",
                how="left"
            )
        
        # Fill NaN values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        # Filter by transaction_ids if provided
        if transaction_ids:
            result = result[result[self.entity].isin(transaction_ids)]
        
        # Add metadata
        result["feature_set_version"] = self.feature_set_version
        result["event_time"] = as_of_time
        
        return result
