"""
Data loaders for preprocessed county data.

This module provides the CleanedDataLoader for loading pre-processed pooled
county data and applying per-experiment Phase 2 preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .preprocessing_utils import Phase2Preprocessor, apply_phase2_preprocessing

logger = logging.getLogger(__name__)


class CleanedDataLoader:
    """
    Loader for pre-cleaned pooled county data.

    This loader is designed to work with data that has undergone Phase 1
    preprocessing (cleaning, feature engineering, label encoding) and
    applies Phase 2 preprocessing (normalization, winsorization, imputation)
    per train/test split to avoid data leakage.
    """

    def __init__(
        self,
        cleaned_data_path: str,
        target_column: str = "SALE_AMOUNT",
        phase2_config: Optional[dict] = None,
    ):
        """
        Initialize loader.

        Args:
            cleaned_data_path: Path to cleaned data parquet file or directory
            target_column: Name of target variable column
            phase2_config: Phase 2 preprocessing configuration:
                {
                    'winsorize': bool,
                    'winsorize_percentile': int,
                    'normalize_continuous': bool,
                    'impute_method': str,  # "median", "mean", "zero", "none"
                }
        """
        self.cleaned_data_path = Path(cleaned_data_path)
        self.target_column = target_column
        self.phase2_config = phase2_config or {}

        # Resolve path (could be file or directory)
        if self.cleaned_data_path.is_dir():
            self.data_file = self.cleaned_data_path / "data.parquet"
            self.metadata_file = self.cleaned_data_path / "metadata.json"
        else:
            self.data_file = self.cleaned_data_path
            self.metadata_file = self.cleaned_data_path.parent / "metadata.json"

        if not self.data_file.exists():
            raise FileNotFoundError(f"Cleaned data file not found: {self.data_file}")

        # Load metadata if available
        self.metadata = self._load_metadata()

        # Cache for loaded data
        self._data_cache: Optional[pd.DataFrame] = None

        logger.info(f"CleanedDataLoader initialized")
        logger.info(f"  Data file: {self.data_file}")
        logger.info(f"  Target column: {self.target_column}")
        if self.metadata:
            logger.info(f"  Dataset version: {self.metadata.get('version', 'unknown')}")
            stats = self.metadata.get('statistics', {})
            final_rows = stats.get('final_rows', 'unknown')
            if isinstance(final_rows, int):
                logger.info(f"  Total rows: {final_rows:,}")
            else:
                logger.info(f"  Total rows: {final_rows}")

    def _load_metadata(self) -> Optional[dict]:
        """Load metadata JSON if available."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return None

    def is_target_log_transformed(self) -> bool:
        """Check if target was log-transformed during Phase 1."""
        if self.metadata:
            return self.metadata.get("preprocessing", {}).get("target_log_transformed", False)
        return False

    def get_continuous_cols(self) -> List[str]:
        """Get list of continuous columns from metadata."""
        if self.metadata:
            return self.metadata.get("columns", {}).get("continuous", [])
        return []

    def get_id_columns(self) -> List[str]:
        """Get list of ID columns (fips, CLIP, etc.)."""
        if self.metadata:
            return self.metadata.get("columns", {}).get("id_columns", ["fips"])
        return ["fips"]

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns."""
        if self.metadata:
            return self.metadata.get("columns", {}).get("all_features", [])
        return []

    def load_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load the full cleaned dataset.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Full cleaned DataFrame
        """
        if use_cache and self._data_cache is not None:
            logger.debug("Using cached data")
            return self._data_cache

        logger.info(f"Loading cleaned data from {self.data_file}")
        df = pd.read_parquet(self.data_file)
        logger.info(f"  Loaded shape: {df.shape}")

        if use_cache:
            self._data_cache = df

        return df

    def load_data_by_indices(self, indices: np.ndarray, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load only specific rows from the dataset.

        Args:
            indices: NumPy array of row indices to load
            max_rows: Optional limit on total rows to read (for debugging/smoke tests).
                     If provided, reads only first max_rows from parquet file directly.

        Returns:
            DataFrame containing only the specified rows
        """
        # Import PyArrow for parquet reading
        import pyarrow.parquet as pq

        if max_rows is not None:
            # Debug mode: just read first N rows from file, ignore indices
            logger.warning(f"DEBUG MODE: Reading only first {max_rows} rows from parquet file (ignoring indices)")

            # Use PyArrow to read only first N rows efficiently
            parquet_file = pq.ParquetFile(self.data_file)

            # Read only first N rows from the first row group(s)
            # This is memory-efficient as it doesn't load the whole file
            table = parquet_file.read_row_groups([0], use_threads=False)
            df = table.to_pandas()

            # If we got more rows than needed, trim
            if len(df) > max_rows:
                df = df.head(max_rows)
            # If we got fewer rows, read additional row groups
            elif len(df) < max_rows:
                row_group_idx = 1
                while len(df) < max_rows and row_group_idx < parquet_file.num_row_groups:
                    additional_table = parquet_file.read_row_groups([row_group_idx], use_threads=False)
                    df = pd.concat([df, additional_table.to_pandas()], ignore_index=True)
                    row_group_idx += 1
                df = df.head(max_rows)

            logger.info(f"  Loaded shape: {df.shape}")
            return df

        # Normal mode: load specific rows by index
        logger.info(f"Loading {len(indices):,} rows by index from {self.data_file}")

        # Ensure indices are sorted for efficient access
        indices = np.sort(indices)

        # Read parquet file with PyArrow
        table = pq.read_table(self.data_file)

        # Take only the specified indices
        subset_table = table.take(indices)

        # Convert to pandas
        df = subset_table.to_pandas()

        logger.info(f"  Loaded shape: {df.shape}")

        return df

    def load_fips_column(self) -> np.ndarray:
        """
        Load only the fips column from the dataset (lightweight).

        Returns a numpy array of fips values (one per row in the parquet file).
        This is much cheaper than loading the full dataset (~134MB for 17.7M rows
        vs ~2.3GB for the full table).
        """
        logger.info(f"Loading fips column from {self.data_file}")
        table = pq.read_table(self.data_file, columns=['fips'])
        fips_arr = table.column('fips').to_numpy()
        logger.info(f"  Loaded {len(fips_arr):,} fips values, {len(np.unique(fips_arr)):,} unique counties")
        return fips_arr

    def get_county_fips_list(self) -> List[int]:
        """Get list of all county FIPS codes in the dataset."""
        df = self.load_data()
        if "fips" in df.columns:
            return sorted(df["fips"].unique().tolist())
        return []

    def get_county_data(
        self,
        fips: int,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target for a single county.

        Args:
            fips: County FIPS code
            df: Optional pre-loaded DataFrame (avoids reloading)

        Returns:
            Tuple of (X, y) for the county
        """
        if df is None:
            df = self.load_data()

        county_df = df[df["fips"] == fips].copy()

        if len(county_df) == 0:
            raise ValueError(f"County {fips} not found in data")

        X, y = self._split_features_target(county_df)
        return X, y

    def get_multiple_counties_data(
        self,
        fips_list: List[int],
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get pooled features and target for multiple counties.

        Args:
            fips_list: List of county FIPS codes
            df: Optional pre-loaded DataFrame

        Returns:
            Tuple of (X, y) for all counties combined
        """
        if df is None:
            df = self.load_data()

        counties_df = df[df["fips"].isin(fips_list)].copy()

        if len(counties_df) == 0:
            raise ValueError(f"No data found for counties: {fips_list}")

        X, y = self._split_features_target(counties_df)
        return X, y

    def _split_features_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).

        Removes ID columns and target from features.
        """
        id_cols = self.get_id_columns()

        # Target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column {self.target_column} not in data")
        y = df[self.target_column].copy()

        # Features (exclude IDs and target)
        exclude_cols = set(id_cols + [self.target_column])
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()

        return X, y

    def prepare_train_test_split(
        self,
        train_fips: List[int],
        test_fips: List[int],
        apply_phase2: bool = True,
        sample_train: Optional[int] = None,
        sample_test: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare train/test split with Phase 2 preprocessing.

        This is the main method for experiment code. It:
        1. Loads the cleaned data
        2. Splits by county FIPS
        3. Optionally samples within splits
        4. Applies Phase 2 preprocessing (fit on train, apply to both)

        Args:
            train_fips: List of training county FIPS codes
            test_fips: List of test county FIPS codes
            apply_phase2: Whether to apply Phase 2 preprocessing
            sample_train: Optional max number of train samples
            sample_test: Optional max number of test samples
            random_state: Random state for sampling

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info(f"Preparing train/test split")
        logger.info(f"  Train counties: {len(train_fips)}")
        logger.info(f"  Test counties: {len(test_fips)}")

        # Load data
        df = self.load_data()

        # Split by county
        X_train, y_train = self.get_multiple_counties_data(train_fips, df)
        X_test, y_test = self.get_multiple_counties_data(test_fips, df)

        logger.info(f"  Train size (before sampling): {len(X_train)}")
        logger.info(f"  Test size (before sampling): {len(X_test)}")

        # Sample if requested
        if sample_train is not None and len(X_train) > sample_train:
            indices = X_train.sample(n=sample_train, random_state=random_state).index
            X_train = X_train.loc[indices]
            y_train = y_train.loc[indices]
            logger.info(f"  Sampled train to {len(X_train)}")

        if sample_test is not None and len(X_test) > sample_test:
            indices = X_test.sample(n=sample_test, random_state=random_state).index
            X_test = X_test.loc[indices]
            y_test = y_test.loc[indices]
            logger.info(f"  Sampled test to {len(X_test)}")

        # Reset indices
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Apply Phase 2 preprocessing
        if apply_phase2 and self.phase2_config:
            logger.info("  Applying Phase 2 preprocessing...")
            continuous_cols = self.get_continuous_cols()
            # Filter to columns that exist in data
            continuous_cols = [c for c in continuous_cols if c in X_train.columns]

            X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                X_train, y_train, X_test, y_test,
                config=self.phase2_config,
                continuous_cols=continuous_cols
            )

        logger.info(f"  Final train shape: {X_train.shape}")
        logger.info(f"  Final test shape: {X_test.shape}")

        return X_train, y_train, X_test, y_test

    def prepare_split_from_indices(
        self,
        df: pd.DataFrame,
        train_indices: pd.Index,
        test_indices: pd.Index,
        apply_phase2: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare train/test split from pre-computed indices.

        Useful when sampling has already been done externally.

        Args:
            df: Full DataFrame (or subset)
            train_indices: Indices for training data
            test_indices: Indices for test data
            apply_phase2: Whether to apply Phase 2 preprocessing

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        train_df = df.loc[train_indices].copy()
        test_df = df.loc[test_indices].copy()

        X_train, y_train = self._split_features_target(train_df)
        X_test, y_test = self._split_features_target(test_df)

        # Reset indices
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Apply Phase 2 preprocessing
        if apply_phase2 and self.phase2_config:
            continuous_cols = self.get_continuous_cols()
            continuous_cols = [c for c in continuous_cols if c in X_train.columns]

            X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                X_train, y_train, X_test, y_test,
                config=self.phase2_config,
                continuous_cols=continuous_cols
            )

        return X_train, y_train, X_test, y_test

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache = None
        logger.debug("Data cache cleared")
