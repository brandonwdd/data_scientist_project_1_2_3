"""Great Expectations Runner A8.1 Data Quality Gate: GE runner for data quality checks"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from platform_sdk.quality.contracts import DataQualityReport
from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class GERunner:
    """Great Expectations data quality runner"""

    def __init__(self, context_root_dir: Optional[str] = None):
        """
        Initialize GE runner
        
        Args:
            context_root_dir: GE context root directory
        """
        self.context_root_dir = context_root_dir
        self.ge_available = self._check_ge_available()

    def _check_ge_available(self) -> bool:
        """Check if Great Expectations is available"""
        try:
            import great_expectations as ge
            return True
        except ImportError:
            logger.warning("Great Expectations not installed. Install with: pip install great-expectations")
            return False

    def run_suite(
        self,
        df,
        suite_name: str,
        expectation_suite: Optional[Dict] = None
    ) -> DataQualityReport:
        """
        Run Great Expectations suite on dataframe
        
        Args:
            df: Pandas DataFrame to validate
            suite_name: Name of expectation suite
            expectation_suite: Optional expectation suite dict
        
        Returns:
            DataQualityReport
        """
        if not self.ge_available:
            logger.warning("GE not available, returning empty report")
            return DataQualityReport()

        try:
            import great_expectations as ge
            from great_expectations.core import ExpectationSuite
            
            # Create GE dataframe
            ge_df = ge.from_pandas(df)
            
            # Load or create suite
            if expectation_suite:
                suite = ExpectationSuite(**expectation_suite)
            else:
                # Try to load from context
                try:
                    context = ge.get_context(context_root_dir=self.context_root_dir)
                    suite = context.get_expectation_suite(suite_name)
                except:
                    logger.warning(f"Suite {suite_name} not found, creating default")
                    suite = self._create_default_suite(ge_df)
            
            # Validate
            validation_result = ge_df.validate(expectation_suite=suite)
            
            # Parse results
            report = self._parse_validation_result(validation_result, df)
            
            return report
        
        except Exception as e:
            logger.error(f"GE validation failed: {e}")
            report = DataQualityReport()
            report.errors.append(str(e))
            return report

    def _create_default_suite(self, ge_df):
        """Create default expectation suite"""
        import great_expectations as ge
        
        # Basic expectations
        ge_df.expect_table_row_count_to_be_between(min_value=1)
        ge_df.expect_table_columns_to_match_ordered_list(ge_df.columns.tolist())
        
        # Check for nulls
        for col in ge_df.columns:
            ge_df.expect_column_values_to_not_be_null(col, mostly=0.95)
        
        return ge_df.get_expectation_suite()

    def _parse_validation_result(
        self,
        validation_result,
        df
    ) -> DataQualityReport:
        """Parse GE validation result into DataQualityReport"""
        report = DataQualityReport()
        
        if not validation_result.success:
            # Calculate missing rates
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_rate = missing_count / len(df) if len(df) > 0 else 0
                report.missing_rate[col] = float(missing_rate)
            
            # Count failed expectations
            failed_expectations = [
                exp for exp in validation_result.results
                if not exp.success
            ]
            
            report.schema_drift = len(failed_expectations)
            
            # Collect errors
            for exp in failed_expectations:
                report.errors.append({
                    "expectation_type": exp.expectation_config.expectation_type,
                    "column": exp.expectation_config.kwargs.get("column"),
                    "message": exp.result.get("message", "Validation failed")
                })
        
        return report

    def save_report(
        self,
        report: DataQualityReport,
        output_path: str
    ):
        """Save data quality report to JSON"""
        report_dict = report.to_dict()
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Data quality report saved to {output_path}")
