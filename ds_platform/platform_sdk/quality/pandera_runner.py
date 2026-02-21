"""Pandera Runner: Pandera schema validation"""

from typing import Dict, Any, Optional
import json
import pandas as pd

from platform_sdk.quality.contracts import DataQualityReport
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class PanderaRunner:
    """Pandera schema validation runner"""

    def __init__(self):
        self.pandera_available = self._check_pandera_available()

    def _check_pandera_available(self) -> bool:
        """Check if Pandera is available"""
        try:
            import pandera
            return True
        except ImportError:
            logger.warning("Pandera not installed. Install with: pip install pandera")
            return False

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Optional[Any] = None,
        schema_dict: Optional[Dict] = None
    ) -> DataQualityReport:
        """
        Validate dataframe against Pandera schema
        
        Args:
            df: DataFrame to validate
            schema: Pandera DataFrameSchema object
            schema_dict: Dictionary representation of schema (alternative)
        
        Returns:
            DataQualityReport
        """
        if not self.pandera_available:
            logger.warning("Pandera not available, returning empty report")
            return DataQualityReport()

        report = DataQualityReport()

        try:
            import pandera as pa
            from pandera.errors import SchemaError

            # Create schema from dict if provided
            if schema_dict and schema is None:
                schema = self._create_schema_from_dict(schema_dict)

            if schema is None:
                logger.warning("No schema provided, creating basic validation")
                schema = self._create_basic_schema(df)

            # Validate
            try:
                validated_df = schema(df, lazy=True)
                logger.info("Schema validation passed")
                
                # Calculate missing rates
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    missing_rate = missing_count / len(df) if len(df) > 0 else 0
                    report.missing_rate[col] = float(missing_rate)
                
            except SchemaError as e:
                logger.error(f"Schema validation failed: {e}")
                report.schema_drift = len(e.schema_errors) if hasattr(e, 'schema_errors') else 1
                
                # Collect errors
                if hasattr(e, 'schema_errors'):
                    for error in e.schema_errors:
                        report.errors.append({
                            "column": error.get("column"),
                            "check": error.get("check"),
                            "message": str(error)
                        })
                else:
                    report.errors.append({"message": str(e)})
                
                # Calculate missing rates
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    missing_rate = missing_count / len(df) if len(df) > 0 else 0
                    report.missing_rate[col] = float(missing_rate)

        except Exception as e:
            logger.error(f"Pandera validation error: {e}")
            report.errors.append(str(e))

        return report

    def _create_schema_from_dict(self, schema_dict: Dict) -> Any:
        """Create Pandera schema from dictionary"""
        import pandera as pa
        
        # Simplified: would need full schema dict structure
        # For now, create basic schema
        columns = {}
        for col_name, col_spec in schema_dict.get("columns", {}).items():
            if col_spec.get("type") == "numeric":
                columns[col_name] = pa.Column(pa.Float, nullable=col_spec.get("nullable", False))
            elif col_spec.get("type") == "categorical":
                columns[col_name] = pa.Column(pa.String, nullable=col_spec.get("nullable", False))
            else:
                columns[col_name] = pa.Column(pa.String, nullable=True)
        
        return pa.DataFrameSchema(columns=columns)

    def _create_basic_schema(self, df: pd.DataFrame) -> Any:
        """Create basic schema from dataframe"""
        import pandera as pa
        
        columns = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                columns[col] = pa.Column(pa.Float, nullable=True)
            else:
                columns[col] = pa.Column(pa.String, nullable=True)
        
        return pa.DataFrameSchema(columns=columns)

    def validate_domain_sets(
        self,
        df: pd.DataFrame,
        domain_sets: Dict[str, list]
    ) -> DataQualityReport:
        """
        Validate domain sets (categorical value constraints)
        
        Args:
            df: DataFrame to validate
            domain_sets: Dictionary mapping column -> allowed values
        
        Returns:
            DataQualityReport with domain_set_violations
        """
        report = DataQualityReport()

        for col, allowed_values in domain_sets.items():
            if col not in df.columns:
                continue
            
            # Check for values outside domain
            invalid_values = df[~df[col].isin(allowed_values + [None, pd.NA])]
            violation_count = len(invalid_values)
            
            if violation_count > 0:
                report.domain_set_violations += violation_count
                report.errors.append({
                    "column": col,
                    "violation_count": violation_count,
                    "invalid_values": invalid_values[col].unique().tolist()[:10]  # First 10
                })

        return report
