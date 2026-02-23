"""Materialize features to platform.online_features."""

from typing import Dict, List, Optional
from datetime import datetime
import json
import pandas as pd

from platform_sdk.db.pg import get_db
from platform_sdk.common.config import Config
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class FeatureMaterializer:
    """Materialize features to online store"""

    def __init__(self, domain: str):
        self.domain = domain
        self.db = get_db()

    def upsert_one(
        self,
        entity_key: str,
        feature_set_version: str,
        features: Dict,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Upsert one entity's features into online_features (for API / admin use).
        """
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            schema = Config.POSTGRES_SCHEMA
            query = text(f"""
                INSERT INTO {schema}.online_features
                    (domain, entity_key, feature_set_version, features, event_time, materialized_at, ttl_seconds)
                VALUES
                    (:domain, :entity_key, :feature_set_version, :features, :event_time, :materialized_at, :ttl_seconds)
                ON CONFLICT (domain, entity_key, feature_set_version)
                DO UPDATE SET
                    features = :features,
                    event_time = :event_time,
                    materialized_at = :materialized_at,
                    ttl_seconds = :ttl_seconds
            """)
            now = datetime.now()
            session.execute(
                query,
                {
                    "domain": self.domain,
                    "entity_key": entity_key,
                    "feature_set_version": feature_set_version,
                    "features": json.dumps(features),
                    "event_time": now,
                    "materialized_at": now,
                    "ttl_seconds": ttl_seconds,
                },
            )
            session.commit()
            logger.debug(f"Upserted online feature: {self.domain} / {entity_key}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert online feature: {e}")
            raise
        finally:
            session.close()

    def materialize_features(
        self,
        features_df: pd.DataFrame,
        feature_set_version: str,
        entity_col: str = "entity_id",
        ttl_seconds: Optional[int] = None
    ):
        """
        Materialize features to platform.online_features
        
        Args:
            features_df: DataFrame with features
            feature_set_version: Feature set version
            entity_col: Entity column name
            ttl_seconds: Optional TTL in seconds
        """
        session = self.db.get_session()
        try:
            from sqlalchemy import text

            for _, row in features_df.iterrows():
                entity_key = f"{entity_col}:{row[entity_col]}"
                
                # Extract features (exclude entity and metadata columns)
                feature_cols = [c for c in features_df.columns 
                               if c not in [entity_col, "event_time", "materialized_at"]]
                features_dict = {col: row[col] for col in feature_cols}

                # Get event_time
                event_time = row.get("event_time", datetime.now())

                schema = Config.POSTGRES_SCHEMA
                query = text(f"""
                    INSERT INTO {schema}.online_features 
                        (domain, entity_key, feature_set_version, features, 
                         event_time, materialized_at, ttl_seconds)
                    VALUES 
                        (:domain, :entity_key, :feature_set_version, :features,
                         :event_time, :materialized_at, :ttl_seconds)
                    ON CONFLICT (domain, entity_key, feature_set_version)
                    DO UPDATE SET
                        features = :features,
                        event_time = :event_time,
                        materialized_at = :materialized_at,
                        ttl_seconds = :ttl_seconds
                """)

                session.execute(
                    query,
                    {
                        "domain": self.domain,
                        "entity_key": entity_key,
                        "feature_set_version": feature_set_version,
                        "features": json.dumps(features_dict),
                        "event_time": event_time,
                        "materialized_at": datetime.now(),
                        "ttl_seconds": ttl_seconds
                    }
                )

            session.commit()
            logger.info(f"Materialized {len(features_df)} features for domain {self.domain}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to materialize features: {e}")
            raise
        finally:
            session.close()

    def cleanup_expired_features(self, feature_set_version: Optional[str] = None):
        """Clean up expired features based on TTL"""
        session = self.db.get_session()
        try:
            from sqlalchemy import text

            schema = Config.POSTGRES_SCHEMA
            query = text(f"""
                DELETE FROM {schema}.online_features
                WHERE domain = :domain
                  AND ttl_seconds IS NOT NULL
                  AND materialized_at + INTERVAL '1 second' * ttl_seconds < NOW()
            """)

            if feature_set_version:
                query = text(f"""
                    DELETE FROM {schema}.online_features
                    WHERE domain = :domain
                      AND feature_set_version = :feature_set_version
                      AND ttl_seconds IS NOT NULL
                      AND materialized_at + INTERVAL '1 second' * ttl_seconds < NOW()
                """)
                result = session.execute(query, {
                    "domain": self.domain,
                    "feature_set_version": feature_set_version
                })
            else:
                result = session.execute(query, {"domain": self.domain})

            session.commit()
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} expired features for domain {self.domain}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup expired features: {e}")
        finally:
            session.close()
