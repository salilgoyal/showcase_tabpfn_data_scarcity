"""
Data filtering utilities for restricting counties and features.

This module provides functionality to filter datasets to specific counties and features,
allowing for modular data subsetting that can be easily enabled/disabled via config.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Set

logger = logging.getLogger(__name__)


class DataFilter:
    """
    Filter for restricting analysis to specific counties and features.

    This class can be configured via experiment config to:
    1. Restrict to specific counties (e.g., those with high feature coverage)
    2. Restrict to specific features (e.g., high-coverage features)

    Filtering can be easily enabled/disabled via config without code changes.
    """

    def __init__(self, filter_config: Optional[dict] = None):
        """
        Initialize DataFilter from config.

        Args:
            filter_config: Dictionary with optional 'counties' and 'features' sections:
                {
                    'counties': {
                        'enabled': True,
                        'source': 'file',  # or 'inline'
                        'file': 'path/to/county_list.csv',  # CSV with 'fips' column
                        'list': [1001, 1003, ...]  # Alternative: inline list
                    },
                    'features': {
                        'enabled': True,
                        'source': 'file',  # or 'inline'
                        'file': 'path/to/feature_list.csv',  # CSV with 'feature_name' column
                        'list': ['feature1', 'feature2', ...]  # Alternative: inline list
                    }
                }
        """
        self.filter_config = filter_config or {}

        # Load allowed counties
        self.allowed_counties: Optional[Set[int]] = None
        if self._is_enabled('counties'):
            self.allowed_counties = self._load_county_list()
            logger.info(f"County filtering enabled: {len(self.allowed_counties)} counties allowed")
        else:
            logger.info("County filtering disabled")

        # Load allowed features
        self.allowed_features: Optional[Set[str]] = None
        if self._is_enabled('features'):
            self.allowed_features = self._load_feature_list()
            logger.info(f"Feature filtering enabled: {len(self.allowed_features)} features allowed")
        else:
            logger.info("Feature filtering disabled")

    def _is_enabled(self, filter_type: str) -> bool:
        """Check if a filter type (counties or features) is enabled."""
        return self.filter_config.get(filter_type, {}).get('enabled', False)

    def _load_county_list(self) -> Set[int]:
        """Load list of allowed counties from config."""
        county_config = self.filter_config.get('counties', {})
        source = county_config.get('source', 'file')

        if source == 'file':
            file_path = county_config.get('file')
            if not file_path:
                raise ValueError("County filter enabled with source='file' but no file path provided")

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"County filter file not found: {file_path}")

            df = pd.read_csv(file_path)
            if 'fips' not in df.columns:
                raise ValueError(f"County filter file must have 'fips' column, found: {df.columns.tolist()}")

            counties = set(df['fips'].astype(int).tolist())
            logger.info(f"Loaded {len(counties)} counties from {file_path}")
            return counties

        elif source == 'inline':
            county_list = county_config.get('list', [])
            if not county_list:
                raise ValueError("County filter enabled with source='inline' but no list provided")
            return set(int(fips) for fips in county_list)

        else:
            raise ValueError(f"Unknown county filter source: {source}")

    def _load_feature_list(self) -> Set[str]:
        """Load list of allowed features from config."""
        feature_config = self.filter_config.get('features', {})
        source = feature_config.get('source', 'file')

        if source == 'file':
            file_path = feature_config.get('file')
            if not file_path:
                raise ValueError("Feature filter enabled with source='file' but no file path provided")

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Feature filter file not found: {file_path}")

            df = pd.read_csv(file_path)
            if 'feature_name' not in df.columns:
                raise ValueError(f"Feature filter file must have 'feature_name' column, found: {df.columns.tolist()}")

            features = set(df['feature_name'].astype(str).tolist())
            logger.info(f"Loaded {len(features)} features from {file_path}")
            return features

        elif source == 'inline':
            feature_list = feature_config.get('list', [])
            if not feature_list:
                raise ValueError("Feature filter enabled with source='inline' but no list provided")
            return set(str(feat) for feat in feature_list)

        else:
            raise ValueError(f"Unknown feature filter source: {source}")

    def filter_county_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter county metadata DataFrame to allowed counties.

        Args:
            df: DataFrame with 'fips' column

        Returns:
            Filtered DataFrame (or original if filtering disabled)
        """
        if self.allowed_counties is None:
            return df

        if 'fips' not in df.columns:
            raise ValueError(f"DataFrame must have 'fips' column for filtering, found: {df.columns.tolist()}")

        original_count = len(df)
        df_filtered = df[df['fips'].isin(self.allowed_counties)].copy()
        filtered_count = len(df_filtered)

        logger.info(f"County metadata filtered: {original_count} -> {filtered_count} counties")

        if filtered_count == 0:
            logger.warning("No counties remain after filtering! Check your filter configuration.")

        return df_filtered

    def filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter feature DataFrame to allowed features.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered DataFrame with only allowed features (or original if filtering disabled)
        """
        if self.allowed_features is None:
            return X

        original_features = set(X.columns)
        common_features = original_features & self.allowed_features

        if len(common_features) == 0:
            logger.warning(
                f"No features remain after filtering! "
                f"Original features: {len(original_features)}, "
                f"Allowed features: {len(self.allowed_features)}, "
                f"Overlap: 0"
            )
            raise ValueError("Feature filtering resulted in zero features. Check your filter configuration.")

        missing_features = self.allowed_features - original_features
        if missing_features:
            logger.warning(
                f"{len(missing_features)} allowed features not found in data: "
                f"{sorted(list(missing_features))[:10]}{'...' if len(missing_features) > 10 else ''}"
            )

        # Preserve column order from allowed_features
        ordered_features = [f for f in self.allowed_features if f in common_features]
        X_filtered = X[ordered_features]

        logger.info(f"Features filtered: {len(original_features)} -> {len(X_filtered.columns)} features")

        return X_filtered

    def is_county_filtering_enabled(self) -> bool:
        """Check if county filtering is enabled."""
        return self.allowed_counties is not None

    def is_feature_filtering_enabled(self) -> bool:
        """Check if feature filtering is enabled."""
        return self.allowed_features is not None
