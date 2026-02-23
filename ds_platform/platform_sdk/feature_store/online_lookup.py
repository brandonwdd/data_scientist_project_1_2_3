"""Query features from platform.online_features."""

from typing import Dict, Optional, List
from datetime import datetime
import json
from platform_sdk.db.pg import get_db
from platform_sdk.common.config import Config

    """Online feature lookup from database"""

    def __init__(self, domain: str):
        self.domain = domain
        self.db = get_db()

    def get_features(
        self,
        entity_key: str,
        feature_set_version: str,
        ttl_hours: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get features for entity from online store
        
        Args:
            entity_key: Entity key (e.g., "user:12345")
            feature_set_version: Feature set version
            ttl_hours: Optional TTL override
        
        Returns:
            Dictionary of features or None if not found/expired
        """
        session = self.db.get_session()
        try:
            from sqlalchemy import text
            
            schema = Config.POSTGRES_SCHEMA
            query = text(f"""
                SELECT features, materialized_at, ttl_seconds
                FROM {schema}.online_features
                WHERE domain = :domain
                  AND entity_key = :entity_key
                  AND feature_set_version = :feature_set_version
            """)
            
            result = session.execute(
                query,
                {
                    "domain": self.domain,
                    "entity_key": entity_key,
                    "feature_set_version": feature_set_version
                }
            ).fetchone()
            
            if result is None:
                return None
            
            features, materialized_at, ttl_seconds = result
            
            # Check TTL
            if ttl_seconds is not None:
                ttl_used = ttl_seconds if ttl_hours is None else ttl_hours * 3600
                age_seconds = (datetime.now() - materialized_at).total_seconds()
                if age_seconds > ttl_used:
                    return None  # Expired
            
            # Parse JSONB
            if isinstance(features, str):
                return json.loads(features)
            return features
        
        finally:
            session.close()

    def batch_get_features(
        self,
        entity_keys: List[str],
        feature_set_version: str
    ) -> Dict[str, Dict]:
        """
        Batch get features for multiple entities
        
        Returns:
            Dictionary mapping entity_key -> features
        """
        results = {}
        for entity_key in entity_keys:
            features = self.get_features(entity_key, feature_set_version)
            if features:
                results[entity_key] = features
        return results
