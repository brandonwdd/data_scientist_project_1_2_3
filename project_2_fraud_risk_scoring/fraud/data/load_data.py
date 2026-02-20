"""
Data Loading Module for Fraud
Loads data from S3 (parquet) and Postgres
"""

import os
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

from fraud.data.contracts import (
    TransactionRecord,
    UserBehaviorRecord,
    RiskSignalRecord,
    UserProfileRecord,
    FraudLabel
)


class DataLoader:
    """Loads fraud-related data from various sources"""

    def __init__(self, s3_bucket: str = "s3://lake/fraud", spark: Optional[SparkSession] = None):
        self.s3_bucket = s3_bucket
        self.spark = spark or self._create_spark_session()

    @staticmethod
    def _create_spark_session() -> SparkSession:
        """Create Spark session for feature engineering"""
        return SparkSession.builder \
            .appName("fraud-feature-engineering") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

    def load_transaction_data(
        self,
        start_date: datetime,
        end_date: datetime,
        transaction_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load transaction data from S3
        
        Args:
            start_date: Start of time window
            end_date: End of time window
            transaction_ids: Optional filter by transaction IDs
        
        Returns:
            DataFrame with transaction records
        """
        path = f"{self.s3_bucket}/bronze/transactions"
        
        # Read parquet files
        df = self.spark.read.parquet(path) \
            .filter(
                (col("event_time") >= start_date) &
                (col("event_time") < end_date)
            )
        
        if transaction_ids:
            df = df.filter(col("transaction_id").isin(transaction_ids))
        
        return df.toPandas()

    def load_behavior_data(
        self,
        start_date: datetime,
        end_date: datetime,
        user_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load user behavior data (transaction history) from S3
        
        Args:
            start_date: Start of time window
            end_date: End of time window
            user_ids: Optional filter by user IDs
        
        Returns:
            DataFrame with behavior events
        """
        path = f"{self.s3_bucket}/bronze/behavior"
        
        df = self.spark.read.parquet(path) \
            .filter(
                (col("event_time") >= start_date) &
                (col("event_time") < end_date)
            )
        
        if user_ids:
            df = df.filter(col("user_id").isin(user_ids))
        
        return df.toPandas()

    def load_risk_signal_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load risk signal data (chargebacks, fraud flags, etc.)"""
        path = f"{self.s3_bucket}/bronze/risk_signals"
        
        df = self.spark.read.parquet(path) \
            .filter(
                (col("event_time") >= start_date) &
                (col("event_time") < end_date)
            )
        
        return df.toPandas()

    def load_user_profile_data(
        self,
        as_of_time: datetime,
        user_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load user profile data (account age, KYC status, etc.)
        
        Note: Profile data is point-in-time (snapshot at as_of_time)
        """
        path = f"{self.s3_bucket}/silver/user_profiles"
        
        df = self.spark.read.parquet(path) \
            .filter(col("snapshot_time") <= as_of_time)
        
        # Get latest snapshot for each user
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number
        
        window = Window.partitionBy("user_id").orderBy(col("snapshot_time").desc())
        df = df.withColumn("rn", row_number().over(window)) \
            .filter(col("rn") == 1) \
            .drop("rn")
        
        if user_ids:
            df = df.filter(col("user_id").isin(user_ids))
        
        return df.toPandas()

    def load_labels(
        self,
        as_of_time: datetime,
        investigation_days: int = 7
    ) -> pd.DataFrame:
        """
        Load fraud labels
        
        Label definition:
        - fraud_label=1: Transaction was confirmed as fraud (chargeback, manual review, etc.)
        - fraud_label=0: Transaction was legitimate
        
        Note: Labels are determined after investigation period (default 7 days)
        """
        path = f"{self.s3_bucket}/silver/labels"
        
        df = self.spark.read.parquet(path) \
            .filter(col("as_of_time") == as_of_time)
        
        return df.toPandas()

    def load_device_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load device fingerprint data"""
        path = f"{self.s3_bucket}/bronze/devices"
        
        df = self.spark.read.parquet(path) \
            .filter(
                (col("event_time") >= start_date) &
                (col("event_time") < end_date)
            )
        
        return df.toPandas()
