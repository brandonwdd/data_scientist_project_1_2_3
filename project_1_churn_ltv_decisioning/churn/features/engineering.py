"""
Feature Engineering
Computes features according to feature_spec.yaml
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, countDistinct, avg, max, min, sum,
    when, datediff, window, lag, lead
)

from churn.data.load_data import DataLoader
from churn.data.contracts import validate_plan_tier, validate_channel


class FeatureEngineer:
    """Feature engineering using Spark"""

    def __init__(self, spark: SparkSession, feature_spec: Dict):
        self.spark = spark
        self.feature_spec = feature_spec
        self.entity = feature_spec.get("entity", "user_id")
        self.feature_set_version = feature_spec.get("feature_set_version", "fs_churn_v1")

    def compute_usage_features(
        self,
        behavior_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute usage features:
        - active_days_30d
        - sessions_7d
        - core_feature_cnt_14d
        """
        spark_df = self.spark.createDataFrame(behavior_df)
        
        # active_days_30d: count distinct days in last 30 days
        window_start = as_of_time - timedelta(days=30)
        active_days = spark_df \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time)
            ) \
            .groupBy(self.entity) \
            .agg(
                countDistinct(
                    col("event_time").cast("date")
                ).alias("active_days_30d")
            )
        
        # sessions_7d: count sessions in last 7 days
        window_start = as_of_time - timedelta(days=7)
        sessions = spark_df \
            .filter(
                (col("event_type") == "session_start") &
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time)
            ) \
            .groupBy(self.entity) \
            .agg(count("*").alias("sessions_7d"))
        
        # core_feature_cnt_14d: count distinct core features in last 14 days
        window_start = as_of_time - timedelta(days=14)
        core_features = spark_df \
            .filter(
                (col("event_type") == "feature_use") &
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time)
            ) \
            .groupBy(self.entity) \
            .agg(
                countDistinct("metadata.feature_name").alias("core_feature_cnt_14d")
            )
        
        # Join all usage features
        result = active_days \
            .join(sessions, on=self.entity, how="outer") \
            .join(core_features, on=self.entity, how="outer") \
            .fillna(0)
        
        return result.toPandas()

    def compute_payment_features(
        self,
        payment_df: pd.DataFrame,
        subscription_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute payment features:
        - payment_fail_30d
        - price_change_flag
        - discount_used_90d
        """
        spark_payment = self.spark.createDataFrame(payment_df)
        spark_sub = self.spark.createDataFrame(subscription_df)
        
        # payment_fail_30d
        window_start = as_of_time - timedelta(days=30)
        payment_fails = spark_payment \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time) &
                (col("status") == "failed")
            ) \
            .groupBy(self.entity) \
            .agg(count("*").alias("payment_fail_30d"))
        
        # price_change_flag: any price change in last 90 days
        window_start = as_of_time - timedelta(days=90)
        price_changes = spark_sub \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time)
            ) \
            .groupBy(self.entity) \
            .agg(
                when(
                    countDistinct("price") > 1, 1
                ).otherwise(0).alias("price_change_flag")
            )
        
        # discount_used_90d
        discount_used = spark_sub \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time) &
                col("coupon_code").isNotNull()
            ) \
            .groupBy(self.entity) \
            .agg(
                when(count("*") > 0, 1).otherwise(0).alias("discount_used_90d")
            )
        
        # Join payment features
        result = payment_fails \
            .join(price_changes, on=self.entity, how="outer") \
            .join(discount_used, on=self.entity, how="outer") \
            .fillna(0)
        
        return result.toPandas()

    def compute_support_features(
        self,
        support_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute support features:
        - tickets_30d
        - avg_resolution_hours_90d
        - high_sev_ticket_30d
        """
        spark_df = self.spark.createDataFrame(support_df)
        
        # tickets_30d
        window_start = as_of_time - timedelta(days=30)
        tickets_30d = spark_df \
            .filter(
                (col("created_at") >= window_start) &
                (col("created_at") < as_of_time)
            ) \
            .groupBy(self.entity) \
            .agg(count("*").alias("tickets_30d"))
        
        # avg_resolution_hours_90d
        window_start = as_of_time - timedelta(days=90)
        avg_resolution = spark_df \
            .filter(
                (col("created_at") >= window_start) &
                (col("created_at") < as_of_time) &
                col("resolution_time_hours").isNotNull()
            ) \
            .groupBy(self.entity) \
            .agg(avg("resolution_time_hours").alias("avg_resolution_hours_90d"))
        
        # high_sev_ticket_30d
        high_sev = spark_df \
            .filter(
                (col("created_at") >= window_start) &
                (col("created_at") < as_of_time) &
                col("severity").isin(["high", "critical"])
            ) \
            .groupBy(self.entity) \
            .agg(count("*").alias("high_sev_ticket_30d"))
        
        # Join support features
        result = tickets_30d \
            .join(avg_resolution, on=self.entity, how="outer") \
            .join(high_sev, on=self.entity, how="outer") \
            .fillna(0)
        
        return result.toPandas()

    def compute_trend_features(
        self,
        behavior_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute trend features:
        - usage_slope_4w
        - recency_days
        """
        spark_df = self.spark.createDataFrame(behavior_df)
        
        # usage_slope_4w: linear trend over last 4 weeks
        window_start = as_of_time - timedelta(days=28)
        # Group by week and count events
        weekly_counts = spark_df \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time)
            ) \
            .withColumn("week", window(col("event_time"), "7 days")) \
            .groupBy(self.entity, "week") \
            .agg(count("*").alias("weekly_count")) \
            .orderBy(self.entity, "week")
        
        # Compute slope (simplified: linear regression)
        # This is a simplified version; full implementation would use regression
        result = weekly_counts \
            .groupBy(self.entity) \
            .agg(
                avg("weekly_count").alias("usage_slope_4w")  # Simplified
            )
        
        # recency_days: days since last activity
        recency = spark_df \
            .filter(col("event_time") < as_of_time) \
            .groupBy(self.entity) \
            .agg(
                max("event_time").alias("last_activity")
            ) \
            .withColumn(
                "recency_days",
                datediff(
                    col("as_of_time"),
                    col("last_activity")
                )
            )
        
        result = result.join(recency, on=self.entity, how="outer")
        
        return result.toPandas()

    def compute_value_features(
        self,
        payment_df: pd.DataFrame,
        subscription_df: pd.DataFrame,
        as_of_time: datetime
    ) -> pd.DataFrame:
        """
        Compute value features:
        - avg_revenue_90d
        - plan_tier
        """
        spark_payment = self.spark.createDataFrame(payment_df)
        spark_sub = self.spark.createDataFrame(subscription_df)
        
        # avg_revenue_90d
        window_start = as_of_time - timedelta(days=90)
        avg_revenue = spark_payment \
            .filter(
                (col("event_time") >= window_start) &
                (col("event_time") < as_of_time) &
                (col("status") == "success")
            ) \
            .groupBy(self.entity) \
            .agg(
                avg("amount").alias("avg_revenue_90d")
            )
        
        # plan_tier: latest plan
        plan_tier = spark_sub \
            .filter(col("event_time") < as_of_time) \
            .groupBy(self.entity) \
            .agg(
                max("event_time").alias("latest_time")
            ) \
            .join(
                spark_sub.select(self.entity, "event_time", "plan"),
                on=[self.entity, "event_time"],
                how="left"
            ) \
            .select(self.entity, col("plan").alias("plan_tier"))
        
        result = avg_revenue.join(plan_tier, on=self.entity, how="outer")
        
        return result.toPandas()

    def compute_all_features(
        self,
        data_loader: DataLoader,
        as_of_time: datetime,
        user_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for given as_of_time
        
        Returns:
            DataFrame with user_id and all features
        """
        # Load raw data
        behavior_df = data_loader.load_behavior_data(
            as_of_time - timedelta(days=90),
            as_of_time
        )
        subscription_df = data_loader.load_subscription_data(
            as_of_time - timedelta(days=90),
            as_of_time
        )
        support_df = data_loader.load_support_data(
            as_of_time - timedelta(days=90),
            as_of_time
        )
        
        # Compute feature groups
        usage_features = self.compute_usage_features(behavior_df, as_of_time)
        payment_features = self.compute_payment_features(
            behavior_df, subscription_df, as_of_time
        )
        support_features = self.compute_support_features(support_df, as_of_time)
        trend_features = self.compute_trend_features(behavior_df, as_of_time)
        value_features = self.compute_value_features(
            behavior_df, subscription_df, as_of_time
        )
        
        # Join all features
        result = usage_features \
            .merge(payment_features, on=self.entity, how="outer") \
            .merge(support_features, on=self.entity, how="outer") \
            .merge(trend_features, on=self.entity, how="outer") \
            .merge(value_features, on=self.entity, how="outer")
        
        # Filter by user_ids if provided
        if user_ids:
            result = result[result[self.entity].isin(user_ids)]
        
        # Add metadata
        result["feature_set_version"] = self.feature_set_version
        result["event_time"] = as_of_time
        
        return result
