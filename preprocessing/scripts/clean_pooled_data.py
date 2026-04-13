#!/usr/bin/env python
"""
Phase 1 Preprocessing: Clean and pool county data.

This script applies Phase 1 preprocessing to all counties, producing a cleaned
dataset ready for per-experiment normalization. Phase 1 includes all preprocessing
steps that do NOT risk data leakage (i.e., don't depend on train/test split).

Usage:
    python preprocessing/scripts/clean_pooled_data.py --config preprocessing/configs/v1_no_onehot.yaml

Output:
    - data.parquet: Cleaned pooled dataset
    - config.yaml: Copy of preprocessing config used
    - metadata.json: Statistics about the dataset
    - preprocessing_log.txt: Detailed log
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filters import DataFilter
from src.data.column_categorizer import (
    ColumnCategorizer,
    get_feature_columns,
    ADMINISTRATIVE_COLUMNS,
)


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / "preprocessing_log.txt"

    # Create logger
    logger = logging.getLogger("preprocessing")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# PHASE 1 PREPROCESSING FUNCTIONS
# ==============================================================================

def load_all_counties(
    county_csvs_dir: Path,
    allowed_counties: Optional[Set[int]],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load all county CSV files into a single DataFrame.

    Args:
        county_csvs_dir: Directory containing county CSV files
        allowed_counties: Set of allowed FIPS codes (None = all)
        logger: Logger instance

    Returns:
        Concatenated DataFrame with all counties
    """
    logger.info(f"Loading counties from {county_csvs_dir}")

    # Find all county files
    county_files = sorted(county_csvs_dir.glob("fips_*.csv"))
    logger.info(f"Found {len(county_files)} county files")

    # Filter to allowed counties if specified
    if allowed_counties is not None:
        county_files = [
            f for f in county_files
            if int(f.stem.replace("fips_", "")) in allowed_counties
        ]
        logger.info(f"After filtering: {len(county_files)} county files")

    if not county_files:
        raise ValueError("No county files to load after filtering")

    # Load each county
    dfs = []
    total_rows = 0

    for i, filepath in enumerate(county_files):
        fips = int(filepath.stem.replace("fips_", ""))

        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["fips"] = fips  # Ensure fips column exists
            dfs.append(df)
            total_rows += len(df)

            if (i + 1) % 10 == 0:
                logger.info(f"  Loaded {i + 1}/{len(county_files)} counties ({total_rows:,} rows)")

        except Exception as e:
            logger.error(f"  Failed to load {filepath}: {e}")

    logger.info(f"Loaded {len(dfs)} counties with {total_rows:,} total rows")

    # Concatenate
    logger.info("Concatenating all counties...")
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {df_all.shape}")

    return df_all


def drop_null_labels(
    df: pd.DataFrame,
    target_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Drop rows where target is null."""
    before = len(df)
    df = df.dropna(subset=[target_column])
    after = len(df)
    logger.info(f"drop_null_labels: {before:,} -> {after:,} rows (dropped {before - after:,})")
    return df.reset_index(drop=True)


def generate_temporal_features(
    df: pd.DataFrame,
    sale_date_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Generate temporal features from sale_date column.

    Creates:
        - sale_year: Year of sale
        - sale_month: Month (1-12)
        - sale_day_of_month: Day of month (1-31)
        - sale_day_of_year: Day of year (1-366)
        - sale_day_of_week: Day of week (0-6, Monday=0)
        - sale_day: Days since Jan 1, 2000
    """
    logger.info(f"Generating temporal features from {sale_date_col}")

    if sale_date_col not in df.columns:
        logger.warning(f"  {sale_date_col} not found, skipping temporal feature generation")
        return df

    # Parse sale date
    sale_date = df[sale_date_col].astype(str).str.zfill(8)
    sale_date = pd.to_datetime(sale_date, format="%Y%m%d", errors="coerce")

    # Generate features
    df["sale_year"] = sale_date.dt.year
    df["sale_month"] = sale_date.dt.month
    df["sale_day_of_month"] = sale_date.dt.day
    df["sale_day_of_year"] = sale_date.dt.dayofyear
    df["sale_day_of_week"] = sale_date.dt.dayofweek

    # Days since Jan 1, 2000
    reference_date = pd.Timestamp("2000-01-01")
    df["sale_day"] = (sale_date - reference_date).dt.days

    null_dates = sale_date.isna().sum()
    if null_dates > 0:
        logger.warning(f"  {null_dates:,} rows have unparseable sale_date")

    logger.info(f"  Generated 6 temporal features")
    return df


def drop_single_value_cols(
    df: pd.DataFrame,
    exclude_cols: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """Drop columns with only one unique value."""
    drop_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1 and col not in exclude_cols
    ]

    if drop_cols:
        logger.info(f"drop_single_value_cols: Dropping {len(drop_cols)} columns")
        logger.debug(f"  Columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    else:
        logger.info("drop_single_value_cols: No columns to drop")

    return df


def drop_mostly_null_cols(
    df: pd.DataFrame,
    share_non_null: float,
    exclude_cols: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """Drop columns with less than share_non_null fraction of non-null values."""
    min_non_null = int(np.floor(share_non_null * len(df)))

    drop_cols = [
        col for col in df.columns
        if df[col].notnull().sum() < min_non_null and col not in exclude_cols
    ]

    if drop_cols:
        logger.info(
            f"drop_mostly_null_cols: Dropping {len(drop_cols)} columns "
            f"(< {share_non_null * 100:.0f}% non-null)"
        )
        logger.debug(f"  Columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    else:
        logger.info("drop_mostly_null_cols: No columns to drop")

    return df


def drop_lowest_ratios(
    df: pd.DataFrame,
    target_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Drop observations in lowest percentile of sales ratios.

    Sales ratio = MARKET_TOTAL_VALUE / SALE_AMOUNT
    Drops observations in the bottom 1 percentile by sale_year.
    """
    if "MARKET_TOTAL_VALUE" not in df.columns or target_column not in df.columns:
        logger.warning("drop_lowest_ratios: Required columns not found, skipping")
        return df

    before = len(df)

    # Calculate ratio
    df = df.copy()
    df["_ratio"] = df["MARKET_TOTAL_VALUE"] / df[target_column]

    if "sale_year" in df.columns:
        # Percentile rank within each year
        def percentile_rank(group):
            return group.rank(pct=True) * 100

        df["_ratio_pct"] = df.groupby("sale_year")["_ratio"].transform(percentile_rank)
        df = df[df["_ratio_pct"] >= 1]
        df = df.drop(columns=["_ratio", "_ratio_pct"])
    else:
        # Global percentile
        df["_ratio_pct"] = pd.qcut(df["_ratio"], q=100, labels=False, duplicates="drop")
        df = df[df["_ratio_pct"] >= 1]
        df = df.drop(columns=["_ratio", "_ratio_pct"])

    after = len(df)
    logger.info(f"drop_lowest_ratios: {before:,} -> {after:,} rows (dropped {before - after:,})")

    return df.reset_index(drop=True)


def drop_highest_ratios(
    df: pd.DataFrame,
    target_column: str,
    logger: logging.Logger,
    assessed_value_column: str = "MARKET_TOTAL_VALUE",
) -> pd.DataFrame:
    """
    Drop observations in highest percentile of assessed-to-sale ratio.

    Ratio = MARKET_TOTAL_VALUE / SALE_AMOUNT (raw, before log transform)
    Drops observations in the top 1 percentile by sale_year.
    Complements drop_lowest_ratios (which drops bottom 1% of the same ratio).
    """
    if assessed_value_column not in df.columns or target_column not in df.columns:
        logger.warning(
            f"drop_highest_ratios: Required columns ({assessed_value_column}, {target_column}) "
            f"not found, skipping"
        )
        return df

    before = len(df)

    df = df.copy()
    df["_ratio"] = df[assessed_value_column] / df[target_column]

    if "sale_year" in df.columns:
        def percentile_rank(group):
            return group.rank(pct=True) * 100

        df["_ratio_pct"] = df.groupby("sale_year")["_ratio"].transform(percentile_rank)
        df = df[df["_ratio_pct"] <= 99]
        df = df.drop(columns=["_ratio", "_ratio_pct"])
    else:
        df["_ratio_bins"] = pd.qcut(df["_ratio"], q=100, labels=False, duplicates="drop")
        df = df[df["_ratio_bins"] < 99]
        df = df.drop(columns=["_ratio", "_ratio_bins"])

    after = len(df)
    logger.info(f"drop_highest_ratios: {before:,} -> {after:,} rows (dropped {before - after:,})")

    return df.reset_index(drop=True)


def drop_repeat_sales(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Drop repeat sales, keeping only the most recent per property per month.

    Deduplicates on (CLIP, sale_year, sale_month), keeping the last occurrence.
    """
    if "CLIP" not in df.columns:
        logger.warning("drop_repeat_sales: CLIP column not found, skipping")
        return df

    before = len(df)

    subset_cols = ["CLIP"]
    if "sale_year" in df.columns and "sale_month" in df.columns:
        subset_cols.extend(["sale_year", "sale_month"])

    df = df.drop_duplicates(subset=subset_cols, keep="last")

    after = len(df)
    logger.info(f"drop_repeat_sales: {before:,} -> {after:,} rows (dropped {before - after:,})")

    return df.reset_index(drop=True)


def label_encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Label encode categorical columns.

    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
        logger: Logger

    Returns:
        Tuple of (encoded DataFrame, encoding_maps)
    """
    encoding_maps = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        logger.debug(f"  Label encoding {col}")

        # Handle missing values
        df[col] = df[col].fillna("_MISSING_")
        df[col] = df[col].astype(str)

        # Encode
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

        # Store mapping
        encoding_maps[col] = {
            label: int(idx) for idx, label in enumerate(le.classes_)
        }

    logger.info(f"label_encode_categoricals: Encoded {len(encoding_maps)} columns")

    return df, encoding_maps


def log_transform_target(
    df: pd.DataFrame,
    target_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Apply log transformation to target column."""
    if target_column not in df.columns:
        logger.error(f"Target column {target_column} not found")
        raise ValueError(f"Target column {target_column} not found")

    # Check for non-positive values
    non_positive = (df[target_column] <= 0).sum()
    if non_positive > 0:
        logger.warning(
            f"log_transform_target: {non_positive:,} rows have non-positive target, will be dropped"
        )
        df = df[df[target_column] > 0].copy()

    df[target_column] = np.log(df[target_column])
    logger.info(f"log_transform_target: Applied log transform to {target_column}")

    return df.reset_index(drop=True)


def filter_features(
    df: pd.DataFrame,
    data_filter: DataFilter,
    keep_cols: List[str],
    target_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply feature filtering, keeping specified columns.

    Args:
        df: DataFrame
        data_filter: DataFilter instance
        keep_cols: Columns to always keep (IDs, target)
        target_column: Target column name
        logger: Logger

    Returns:
        Filtered DataFrame
    """
    if not data_filter.is_feature_filtering_enabled():
        logger.info("Feature filtering not enabled, keeping all columns")
        return df

    # Get columns that would be kept by the filter
    feature_cols = [c for c in df.columns if c not in keep_cols and c != target_column]
    feature_df = df[feature_cols]

    # Apply filter
    filtered_feature_df = data_filter.filter_features(feature_df)

    # Reconstruct with keep_cols
    keep_existing = [c for c in keep_cols if c in df.columns]
    final_cols = keep_existing + [target_column] + list(filtered_feature_df.columns)
    final_cols = [c for c in final_cols if c in df.columns]  # Dedupe and validate

    df_filtered = df[final_cols]
    logger.info(f"filter_features: {len(df.columns)} -> {len(df_filtered.columns)} columns")

    return df_filtered


# ==============================================================================
# MAIN PREPROCESSING PIPELINE
# ==============================================================================

def run_phase1_preprocessing(config: dict, logger: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete Phase 1 preprocessing pipeline.

    Args:
        config: Preprocessing configuration
        logger: Logger instance

    Returns:
        Tuple of (preprocessed DataFrame, metadata dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 1 PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    # Extract config sections
    data_config = config["data"]
    filter_config = config.get("data_filtering", {})
    feature_config = config.get("features", {})
    step_config = config.get("phase1_steps", {})

    target_column = data_config["target_column"]
    county_csvs_dir = Path(data_config["county_csvs_dir"])

    # Initialize metadata
    metadata = {
        "version": config["version"],
        "description": config["description"],
        "created_at": datetime.now().isoformat(),
        "config_file": None,  # Will be set later
        "statistics": {},
    }

    # Initialize data filter
    logger.info("Initializing data filter...")
    data_filter = DataFilter(filter_config)

    # Get allowed counties
    allowed_counties = data_filter.allowed_counties

    # Load all counties
    logger.info("-" * 40)
    df = load_all_counties(county_csvs_dir, allowed_counties, logger)
    metadata["statistics"]["initial_rows"] = len(df)
    metadata["statistics"]["initial_columns"] = len(df.columns)
    metadata["statistics"]["n_counties"] = df["fips"].nunique() if "fips" in df.columns else 0

    # --- Apply Phase 1 preprocessing steps ---

    # 1. Drop null labels
    if step_config.get("drop_null_labels", True):
        logger.info("-" * 40)
        df = drop_null_labels(df, target_column, logger)

    # 2. Generate temporal features (before other drops so sale_year is available)
    if step_config.get("generate_temporal_features", True):
        logger.info("-" * 40)
        df = generate_temporal_features(df, "sale_date", logger)

    # Define columns to exclude from dropping
    # Only keep ID columns and target - administrative columns SHOULD be dropped
    keep_id_cols = step_config.get("keep_id_columns", ["fips", "CLIP"])
    exclude_from_drops = keep_id_cols + [target_column]

    # 3. Drop single value columns
    if step_config.get("drop_single_value_cols", True):
        logger.info("-" * 40)
        df = drop_single_value_cols(df, exclude_from_drops, logger)

    # 4. Drop mostly null columns
    if step_config.get("drop_mostly_null_cols", True):
        logger.info("-" * 40)
        share_non_null = step_config.get("share_non_null", 0.5)
        df = drop_mostly_null_cols(df, share_non_null, exclude_from_drops, logger)

    # 5. Drop lowest ratios
    if step_config.get("drop_lowest_ratios", True):
        logger.info("-" * 40)
        df = drop_lowest_ratios(df, target_column, logger)

    # 5b. Drop highest ratios
    if step_config.get("drop_highest_ratios", False):
        logger.info("-" * 40)
        df = drop_highest_ratios(df, target_column, logger)

    # 6. Drop repeat sales
    if step_config.get("drop_repeat_sales", True):
        logger.info("-" * 40)
        df = drop_repeat_sales(df, logger)

    # 7. Apply feature filtering
    logger.info("-" * 40)
    df = filter_features(df, data_filter, keep_id_cols, target_column, logger)

    # 8. Identify column types using column categorizer
    logger.info("-" * 40)
    logger.info("Categorizing columns...")
    feature_cols_dict = get_feature_columns(df, feature_config, target_column)

    categorical_cols = feature_cols_dict["categorical_cols"]
    continuous_cols = feature_cols_dict["continuous_cols"]
    binary_cols = feature_cols_dict["binary_cols"]

    # 9. Handle categoricals
    categorical_handling = step_config.get("categorical_handling", "label_encode")
    logger.info("-" * 40)
    logger.info(f"Handling categoricals: {categorical_handling}")

    encoding_maps = {}
    if categorical_handling == "label_encode":
        df, encoding_maps = label_encode_categoricals(df, categorical_cols, logger)
    elif categorical_handling == "drop":
        df = df.drop(columns=[c for c in categorical_cols if c in df.columns])
        logger.info(f"Dropped {len(categorical_cols)} categorical columns")
    elif categorical_handling == "keep_string":
        logger.info("Keeping categorical columns as strings")
    else:
        logger.warning(f"Unknown categorical_handling: {categorical_handling}")

    # 10. Log transform target
    if step_config.get("log_transform_target", True):
        logger.info("-" * 40)
        df = log_transform_target(df, target_column, logger)

    # --- Finalize ---
    logger.info("-" * 40)
    logger.info("Finalizing...")

    # Reset index
    df = df.reset_index(drop=True)

    # Update metadata
    metadata["statistics"]["final_rows"] = len(df)
    metadata["statistics"]["final_columns"] = len(df.columns)
    metadata["statistics"]["n_counties_final"] = df["fips"].nunique() if "fips" in df.columns else 0

    # Column type info
    metadata["columns"] = {
        "target": target_column,
        "id_columns": [c for c in keep_id_cols if c in df.columns],
        "continuous": [c for c in continuous_cols if c in df.columns],
        "binary": [c for c in binary_cols if c in df.columns],
        "categorical": [c for c in categorical_cols if c in df.columns],
        "all_features": [c for c in df.columns if c not in keep_id_cols and c != target_column],
    }

    metadata["preprocessing"] = {
        "target_log_transformed": step_config.get("log_transform_target", True),
        "categorical_handling": categorical_handling,
        "encoding_maps": encoding_maps,
    }

    # Log summary
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Counties: {metadata['statistics']['n_counties_final']}")
    logger.info(f"Features: {len(metadata['columns']['all_features'])}")
    logger.info(f"Target log-transformed: {metadata['preprocessing']['target_log_transformed']}")

    return df, metadata


def save_outputs(
    df: pd.DataFrame,
    metadata: dict,
    config: dict,
    config_path: str,
    logger: logging.Logger
) -> Path:
    """
    Save preprocessed data and metadata.

    Args:
        df: Preprocessed DataFrame
        metadata: Metadata dictionary
        config: Config dictionary
        config_path: Path to original config file
        logger: Logger

    Returns:
        Output directory path
    """
    output_config = config["output"]
    base_dir = Path(output_config["base_dir"])
    version = config["version"]

    # Create output directory
    output_dir = base_dir / version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving outputs to {output_dir}")

    # Save data
    output_format = output_config.get("format", "parquet")
    if output_format == "parquet":
        compression = output_config.get("compression", "snappy")
        data_path = output_dir / "data.parquet"
        df.to_parquet(data_path, compression=compression, index=False)
        logger.info(f"  Saved data to {data_path} ({data_path.stat().st_size / 1e9:.2f} GB)")
    elif output_format == "feather":
        data_path = output_dir / "data.feather"
        df.to_feather(data_path)
        logger.info(f"  Saved data to {data_path}")
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    # Save config copy
    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"  Saved config copy to {config_copy_path}")

    # Update metadata with paths
    metadata["config_file"] = str(config_copy_path)
    metadata["data_file"] = str(data_path)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to {metadata_path}")

    return output_dir


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Preprocessing: Clean and pool county data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing config file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create output directory for logging
    output_dir = Path(config["output"]["base_dir"]) / config["version"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    logger.info(f"Config: {config_path}")
    logger.info(f"Version: {config['version']}")

    try:
        # Run preprocessing
        df, metadata = run_phase1_preprocessing(config, logger)

        # Save outputs
        save_outputs(df, metadata, config, str(config_path), logger)

        logger.info("SUCCESS!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
