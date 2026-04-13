"""
Data loaders for county CSV files.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Add evelyn_files to path for Evelyn's preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'evelyn_files'))

logger = logging.getLogger(__name__)


class CountyDataLoader:
    """Loads and preprocesses county data."""

    def __init__(
        self,
        county_csvs_dir: str,
        target_column: str,
        preprocessing_config: Optional[dict] = None,
        # DEPRECATED: Old interface for backward compatibility
        use_evelyn_preprocessing: bool = None,
        include_property_chars: bool = None,
        property_chars_only: bool = None
    ):
        """
        Initialize loader.

        Args:
            county_csvs_dir: Directory containing county CSV files
            target_column: Name of target variable column
            preprocessing_config: NEW format - Full preprocessing configuration:
                {
                    'features': {
                        'property_chars': bool,
                        'census_bg': bool,
                        'census_tract': bool,
                        'assessed_value': bool,
                        'geographic': bool,
                        'temporal': bool,
                    },
                    'steps': {
                        'winsorize': bool,
                        'log_transform_target': bool,
                        'normalize_continuous': bool,
                        ...
                    }
                }
                If None, uses original preprocessing (no Evelyn pipeline).
            use_evelyn_preprocessing: DEPRECATED - Use preprocessing_config instead
            include_property_chars: DEPRECATED - Use preprocessing_config instead
            property_chars_only: DEPRECATED - Use preprocessing_config instead
        """
        self.county_csvs_dir = Path(county_csvs_dir)
        self.target_column = target_column

        # Handle old vs new config format
        if preprocessing_config is not None:
            # New format
            self.preprocessing_config = preprocessing_config
            self.use_evelyn_preprocessing = True
        elif use_evelyn_preprocessing is not None:
            # Old format - convert to new format
            logger.warning(
                "Old preprocessing config format detected. "
                "Please migrate to new 'preprocessing_config' format."
            )
            self.preprocessing_config = self._convert_old_config(
                use_evelyn_preprocessing,
                include_property_chars or False,
                property_chars_only or False
            )
            self.use_evelyn_preprocessing = use_evelyn_preprocessing
        else:
            # No preprocessing
            self.preprocessing_config = None
            self.use_evelyn_preprocessing = False

        if not self.county_csvs_dir.exists():
            raise ValueError(f"County data directory not found: {county_csvs_dir}")

        # Import Evelyn preprocessing if needed
        if self.use_evelyn_preprocessing:
            from .evelyn_preprocessing import load_and_prepare_data
            self._load_and_prepare_data = load_and_prepare_data

            if self.preprocessing_config:
                logger.info(
                    f"Using Evelyn's modular preprocessing pipeline with "
                    f"{len([k for k, v in self.preprocessing_config.get('features', {}).items() if v])} "
                    f"feature categories enabled"
                )

    def _convert_old_config(
        self,
        use_evelyn: bool,
        include_property_chars: bool,
        property_chars_only: bool
    ) -> dict:
        """
        Convert old config flags to new format.

        Args:
            use_evelyn: Whether to use Evelyn preprocessing
            include_property_chars: Whether to include property characteristics
            property_chars_only: Whether to use ONLY property characteristics

        Returns:
            New format config dict
        """
        if not use_evelyn:
            return None

        # Determine feature flags from old logic
        if property_chars_only:
            # Property chars only mode
            feature_config = {
                'property_chars': True,
                'census_bg': False,
                'census_tract': False,
                'assessed_value': False,
                'geographic': False,
                'temporal': True,  # Temporal always generated
            }
        elif include_property_chars:
            # Full features mode
            feature_config = {
                'property_chars': True,
                'census_bg': True,
                'census_tract': False,
                'assessed_value': True,
                'geographic': True,
                'temporal': True,
            }
        else:
            # Minimal mode (assessed value + census only)
            feature_config = {
                'property_chars': False,
                'census_bg': True,
                'census_tract': False,
                'assessed_value': True,
                'geographic': False,
                'temporal': True,
            }

        # Default step config (all steps enabled as in old code)
        step_config = {
            'drop_null_labels': True,
            'drop_single_value_cols': True,
            'drop_mostly_null_cols': True,
            'share_non_null': 0.5,
            'drop_lowest_ratios': True,
            'drop_repeat_sales': True,
            'generate_temporal_features': True,
            'one_hot_encode': True,
            'winsorize': True,
            'winsorize_percentile': 1,
            'log_transform_target': True,
            'normalize_continuous': True,
            'normalize_binary': False,
            'impute_method': 'median',
            'mice_iterations': 3,
        }

        return {
            'features': feature_config,
            'steps': step_config,
        }

    def is_log_transformed(self) -> bool:
        """
        Check if target is log-transformed based on config.

        Returns:
            True if target is log-transformed, False otherwise
        """
        if not self.use_evelyn_preprocessing:
            return False
        if not self.preprocessing_config:
            return False
        return self.preprocessing_config.get('steps', {}).get('log_transform_target', False)

    def get_county_filepath(self, fips: int) -> Path:
        """Get file path for a county."""
        filepath = self.county_csvs_dir / f"fips_{fips}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"County file not found: {filepath}")
        return filepath

    def load_county(self, fips: int, drop_missing_target: bool = True) -> pd.DataFrame:
        """
        Load a single county's data.

        Args:
            fips: FIPS code
            drop_missing_target: Drop rows with missing target values

        Returns:
            DataFrame with county data
        """
        filepath = self.get_county_filepath(fips)
        logger.debug(f"Loading county {fips} from {filepath}")

        df = pd.read_csv(filepath, low_memory=False)

        # Drop missing targets
        if drop_missing_target:
            original_len = len(df)
            df = df.dropna(subset=[self.target_column])
            if len(df) < original_len:
                logger.debug(
                    f"County {fips}: Dropped {original_len - len(df)} rows "
                    f"with missing target"
                )

        return df

    def load_multiple_counties(
        self,
        fips_list: List[int],
        drop_missing_target: bool = True
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple counties.

        Args:
            fips_list: List of FIPS codes
            drop_missing_target: Drop rows with missing target values

        Returns:
            Concatenated DataFrame
        """
        dfs = []
        for fips in fips_list:
            try:
                df = self.load_county(fips, drop_missing_target)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading county {fips}: {e}")

        if not dfs:
            raise ValueError("No counties successfully loaded")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(
            f"Loaded {len(fips_list)} counties: "
            f"{len(combined)} total samples"
        )

        return combined

    def preprocess_for_training(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training.

        Automatically uses Evelyn's preprocessing if use_evelyn_preprocessing=True,
        otherwise uses the original simple preprocessing.

        Args:
            df: Raw DataFrame
            exclude_columns: Columns to exclude from features (in addition to target)
                            (only used for original preprocessing)

        Returns:
            Tuple of (X, y)

        Note:
            When using Evelyn's preprocessing, y is LOG-TRANSFORMED.
            Make sure to pass log_transformed=True to compute_metrics().
        """
        if self.use_evelyn_preprocessing:
            return self._preprocess_evelyn(df)
        else:
            return self._preprocess_original(df, exclude_columns)

    def _preprocess_original(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Original simple preprocessing (no transformations).

        Args:
            df: Raw DataFrame
            exclude_columns: Columns to exclude from features

        Returns:
            Tuple of (X, y)
        """
        # Default columns to exclude
        default_exclude = [
            self.target_column,
            '',  # Unnamed index column
            'CLIP',
            'fips',
            'CENSUS_ID',
            'address',
            'sale_date',
            'OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID',
            'PREVIOUS_CLIP',
            'tract_id',
            'block_group_id',
        ]

        if exclude_columns:
            default_exclude.extend(exclude_columns)

        # Extract target
        y = df[self.target_column].copy()

        # Select features
        feature_cols = [col for col in df.columns if col not in default_exclude]

        X = df[feature_cols].copy()

        # Drop object/string columns
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.debug(f"Dropping {len(object_cols)} object columns: {object_cols}")
            X = X.drop(columns=object_cols)

        # Drop columns with all NaNs
        na_cols = X.columns[X.isna().all()].tolist()
        if na_cols:
            logger.debug(f"Dropping {len(na_cols)} all-NaN columns")
            X = X.drop(columns=na_cols)

        # Fill remaining NaNs with column median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

        logger.debug(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y

    def _preprocess_evelyn(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Evelyn's preprocessing pipeline using new modular system.

        Args:
            df: Raw DataFrame

        Returns:
            Tuple of (X, y) where y may be LOG-TRANSFORMED depending on config

        Note:
            This now uses the new modular preprocessing system with
            fine-grained control via preprocessing_config.
        """
        logger.debug("Applying Evelyn's modular preprocessing pipeline")

        # Save dataframe to temp file for load_and_prepare_data
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
            df.to_csv(temp_path, index=False)

        try:
            # Use new modular preprocessing function
            X_train, y_train, X_test, y_test, _ = self._load_and_prepare_data(
                data_path=temp_path,
                feature_config=self.preprocessing_config['features'],
                step_config=self.preprocessing_config['steps'],
                cbg_column='block_group_id'
            )

            # Combine train and test back together
            X = pd.concat([X_train, X_test], ignore_index=True)
            y = pd.concat([y_train, y_test], ignore_index=True)

            # Reset index
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            log_transformed = self.preprocessing_config['steps'].get('log_transform_target', True)
            logger.debug(
                f"Evelyn preprocessing complete: {X.shape[0]} samples, "
                f"{X.shape[1]} features (target log-transformed: {log_transformed})"
            )

            return X, y

        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def check_feature_consistency(self, fips_list: List[int]) -> dict:
        """
        Check if counties have consistent features.

        Args:
            fips_list: List of FIPS codes to check

        Returns:
            Dictionary with consistency info
        """
        logger.info(f"Checking feature consistency across {len(fips_list)} counties")

        all_features = {}

        for fips in fips_list:
            try:
                df = self.load_county(fips, drop_missing_target=False)
                all_features[fips] = set(df.columns)
            except Exception as e:
                logger.warning(f"Could not load county {fips}: {e}")

        if not all_features:
            return {"consistent": False, "error": "No counties loaded"}

        # Find common features
        common_features = set.intersection(*all_features.values())
        all_unique_features = set.union(*all_features.values())

        consistent = len(common_features) == len(all_unique_features)

        result = {
            "consistent": consistent,
            "num_counties_checked": len(all_features),
            "num_common_features": len(common_features),
            "num_total_unique_features": len(all_unique_features),
            "missing_features_by_county": {}
        }

        if not consistent:
            # Find which counties are missing which features
            for fips, features in all_features.items():
                missing = all_unique_features - features
                if missing:
                    result["missing_features_by_county"][fips] = list(missing)

        return result
