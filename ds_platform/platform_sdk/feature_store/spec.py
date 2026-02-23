"""Parse and validate feature_spec.yaml."""

import yaml
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class FeatureSpec:
    """Feature specification parser"""

    def __init__(self, spec_path: Optional[str] = None, spec_dict: Optional[Dict] = None):
        """
        Initialize feature spec
        
        Args:
            spec_path: Path to feature_spec.yaml
            spec_dict: Feature spec dictionary (alternative)
        """
        if spec_path:
            with open(spec_path, "r") as f:
                self.spec = yaml.safe_load(f)
        elif spec_dict:
            self.spec = spec_dict
        else:
            raise ValueError("Either spec_path or spec_dict must be provided")

        self.entity = self.spec.get("entity", "entity_id")
        self.feature_set_version = self.spec.get("feature_set_version", "v1")
        self.features = self.spec.get("features", [])
        self.domain_sets = self.spec.get("domain_sets", {})
        self.feature_groups = self.spec.get("feature_groups", {})

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return [f["name"] for f in self.features]

    def get_feature_by_name(self, name: str) -> Optional[Dict]:
        """Get feature definition by name"""
        for f in self.features:
            if f["name"] == name:
                return f
        return None

    def get_features_by_group(self, group: str) -> List[str]:
        """Get feature names in a group"""
        return self.feature_groups.get(group, [])

    def validate_feature_values(
        self,
        feature_name: str,
        values: Any
    ) -> bool:
        """
        Validate feature values against domain sets
        
        Returns:
            True if valid, False otherwise
        """
        feature_def = self.get_feature_by_name(feature_name)
        if not feature_def:
            return False

        # Check domain set if categorical
        if feature_def.get("type") == "categorical":
            categories = feature_def.get("categories", [])
            if categories:
                # Check if values are in categories
                if isinstance(values, (list, pd.Series)):
                    return all(v in categories for v in values)
                else:
                    return values in categories

        # Check domain_sets
        if feature_name in self.domain_sets:
            allowed = self.domain_sets[feature_name]
            if isinstance(values, (list, pd.Series)):
                return all(v in allowed for v in values)
            else:
                return values in allowed

        return True

    def get_refresh_frequency(self, feature_name: str) -> Optional[str]:
        """Get refresh frequency for feature"""
        feature_def = self.get_feature_by_name(feature_name)
        return feature_def.get("refresh_frequency") if feature_def else None

    def get_ttl_hours(self, feature_name: str) -> Optional[int]:
        """Get TTL in hours for feature"""
        feature_def = self.get_feature_by_name(feature_name)
        ttl_hours = feature_def.get("ttl_hours")
        if ttl_hours:
            return ttl_hours
        # Default TTL based on refresh frequency
        refresh = self.get_refresh_frequency(feature_name)
        if refresh == "hourly":
            return 24
        elif refresh == "daily":
            return 48
        return None
