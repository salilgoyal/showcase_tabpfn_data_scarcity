"""
Split strategies for test and train set construction.

This module implements strategies for creating test and training sets
based on size-stratified county selection and temporal splits.

Key concepts:
- Test set: Fixed set of counties with temporal split (recent sales = test)
- Train set: Various strategies for constructing training data from remaining data
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TestSetResult:
    """Results from test set creation."""
    test_counties: List[int]  # FIPS codes of test counties
    test_indices: np.ndarray  # Row indices in full dataset for test
    train_pool_indices: np.ndarray  # Row indices available for training (historical from test counties)
    county_info: Dict[int, Dict]  # Per-county statistics
    size_buckets: Dict[str, List[int]]  # Counties in each size bucket
    metadata: Dict[str, Any]


@dataclass
class TrainSetResult:
    """Results from train set creation."""
    train_indices: np.ndarray  # Final training row indices
    source_breakdown: Dict[str, int]  # How many samples from each source
    county_distribution: Dict[int, int]  # Samples per county
    metadata: Dict[str, Any]


# ==============================================================================
# TEST SET CREATION
# ==============================================================================

def create_test_set(
    df: pd.DataFrame,
    config: Dict,
    fips_column: str = "fips",
    date_column: str = "sale_date",
    random_seed: int = 42,
    split_seed: int = None,
    target_column: str = "SALE_AMOUNT",
    baseline_column: str = "CALCULATED_TOTAL_VALUE"
) -> TestSetResult:
    """
    Create a test set based on size-stratified county selection with temporal or random split.

    Args:
        df: Full preprocessed DataFrame
        config: Test set configuration dict
        fips_column: Column name for county FIPS codes
        date_column: Column name for sale dates (can be numeric or datetime)
        random_seed: Random seed for county selection
        split_seed: Random seed for within-county random split (only used when
                    random_split.enabled is true; defaults to random_seed if not provided)
        target_column: Column name for target variable (for baseline ratio calculation)
        baseline_column: Column name for baseline predictions (typically CALCULATED_TOTAL_VALUE)

    Returns:
        TestSetResult with test indices, train pool indices, and metadata
    """
    if split_seed is None:
        split_seed = random_seed

    np.random.seed(random_seed)

    logger.info("Creating test set with size-stratified county selection")

    # Get county sizes
    county_sizes = df.groupby(fips_column).size().to_dict()

    # Parse size buckets from config
    size_buckets_config = config.get("county_selection", {}).get("size_buckets", [])
    if not size_buckets_config:
        # Default buckets based on actual distribution
        size_buckets_config = [
            {"name": "tiny", "min_rows": 50, "max_rows": 100, "n_counties": 5},
            {"name": "small", "min_rows": 100, "max_rows": 500, "n_counties": 8},
            {"name": "medium", "min_rows": 500, "max_rows": 5000, "n_counties": 10},
            {"name": "large", "min_rows": 5000, "max_rows": 50000, "n_counties": 8},
            {"name": "xlarge", "min_rows": 50000, "max_rows": None, "n_counties": 5},
        ]

    # Select counties for each bucket
    selected_counties = []
    size_buckets = {}

    for bucket in size_buckets_config:
        bucket_name = bucket["name"]
        min_rows = bucket.get("min_rows", 0)
        max_rows = bucket.get("max_rows", float('inf'))
        if max_rows is None:
            max_rows = float('inf')
        n_counties = bucket.get("n_counties", 5)

        # Find eligible counties
        eligible = [
            fips for fips, size in county_sizes.items()
            if min_rows <= size < max_rows
        ]

        # Randomly select counties
        n_select = min(n_counties, len(eligible))
        if n_select < n_counties:
            logger.warning(f"Bucket {bucket_name}: Only {len(eligible)} counties available, requested {n_counties}")

        selected = list(np.random.choice(eligible, size=n_select, replace=False))
        selected_counties.extend(selected)
        size_buckets[bucket_name] = selected

        logger.info(f"Bucket {bucket_name} ({min_rows}-{max_rows}): selected {len(selected)} counties")

    # Determine split strategy
    random_split_config = config.get("random_split", {})
    use_random_split = random_split_config.get("enabled", False)

    # Get temporal split configuration (used when random_split not enabled)
    temporal_config = config.get("temporal_split", {})
    test_percentile = temporal_config.get("test_percentile", 50)

    # For random split
    test_fraction = random_split_config.get("test_fraction", 0.2)
    split_rng = np.random.RandomState(split_seed)

    # Split each test county
    test_indices = []
    train_pool_indices = []
    county_info = {}

    for fips in selected_counties:
        county_mask = df[fips_column] == fips
        county_df = df[county_mask]
        county_idx = county_df.index.values

        if use_random_split:
            # Random split: sample test_fraction of rows uniformly at random
            n_test = max(1, int(len(county_idx) * test_fraction))
            test_idx = split_rng.choice(county_idx, size=n_test, replace=False)
            train_idx = np.setdiff1d(county_idx, test_idx)
            extra_info = {"split_seed": split_seed, "test_fraction": test_fraction}
        else:
            # Temporal split: most recent test_percentile% by date = test
            if date_column in county_df.columns:
                dates = county_df[date_column].values
            else:
                dates = county_df.get("sale_year", county_df.get("SALE_YEAR", np.arange(len(county_df)))).values

            cutoff = np.percentile(dates, 100 - test_percentile)
            test_mask = dates >= cutoff
            train_mask = dates < cutoff
            test_idx = county_idx[test_mask]
            train_idx = county_idx[train_mask]
            extra_info = {"temporal_cutoff": float(cutoff)}

        test_indices.extend(test_idx)
        train_pool_indices.extend(train_idx)

        county_info[fips] = {
            "total_rows": len(county_df),
            "test_rows": len(test_idx),
            "train_pool_rows": len(train_idx),
            "size_bucket": next(
                (name for name, counties in size_buckets.items() if fips in counties),
                "unknown"
            ),
            **extra_info,
        }

        logger.debug(f"County {fips}: {len(test_idx)} test, {len(train_idx)} train pool")

    test_indices = np.array(test_indices)
    train_pool_indices = np.array(train_pool_indices)

    metadata = {
        "version": config.get("version", "unknown"),
        "description": config.get("description", ""),
        "n_test_counties": len(selected_counties),
        "n_test_samples": len(test_indices),
        "n_train_pool_samples": len(train_pool_indices),
        "random_seed": random_seed,
        "split_seed": split_seed,
    }
    if use_random_split:
        metadata["split_method"] = "random"
        metadata["test_fraction"] = test_fraction
    else:
        metadata["split_method"] = "temporal"
        metadata["test_percentile"] = test_percentile

    logger.info(f"Test set created: {len(selected_counties)} counties, "
                f"{len(test_indices)} test samples, {len(train_pool_indices)} train pool samples")

    return TestSetResult(
        test_counties=selected_counties,
        test_indices=test_indices,
        train_pool_indices=train_pool_indices,
        county_info=county_info,
        size_buckets=size_buckets,
        metadata=metadata
    )


# ==============================================================================
# TRAIN SET CREATION
# ==============================================================================

def create_train_set(
    df: pd.DataFrame,
    config: Dict,
    test_result: TestSetResult,
    fips_column: str = "fips",
    random_seed: int = 42
) -> TrainSetResult:
    """
    Create a training set based on the specified strategy.

    Args:
        df: Full preprocessed DataFrame
        config: Train set configuration dict
        test_result: TestSetResult from create_test_set
        fips_column: Column name for county FIPS codes
        random_seed: Random seed for reproducibility

    Returns:
        TrainSetResult with training indices and metadata
    """
    np.random.seed(random_seed)

    logger.info(f"Creating train set with version: {config.get('version', 'unknown')}")

    data_sources = config.get("data_sources", {})
    sampling_config = config.get("sampling", {})

    train_indices = []
    source_breakdown = {"test_counties_historical": 0, "external_counties": 0}
    county_distribution = {}

    # Get max_samples for calculating target sizes
    max_samples = sampling_config.get("max_samples", None)
    max_samples_per_county = sampling_config.get("max_samples_per_county", None)

    # Source 1: Historical data from test counties
    test_hist_config = data_sources.get("test_counties_historical", {})
    if test_hist_config.get("enabled", False):
        fraction = test_hist_config.get("fraction", 1.0)
        pool = test_result.train_pool_indices

        # Calculate target based on fraction of max_samples (not pool size)
        if max_samples:
            target_samples = int(max_samples * fraction)
        else:
            target_samples = int(len(pool) * fraction)

        target_samples = min(target_samples, len(pool))

        # Apply per-county cap if specified
        if max_samples_per_county and target_samples > 0:
            # Group pool indices by county
            pool_counties = {}
            for idx in pool:
                fips = df.iloc[idx][fips_column]
                if fips not in pool_counties:
                    pool_counties[fips] = []
                pool_counties[fips].append(idx)

            # Directly add samples from each county up to target
            # Add all available samples (up to cap) until we hit target_samples
            selected = []
            county_order = list(pool_counties.keys())
            np.random.shuffle(county_order)  # Random order for fairness

            for fips in county_order:
                if len(selected) >= target_samples:
                    break

                county_indices = pool_counties[fips]
                # Take min(county_size, cap, remaining_needed)
                remaining_needed = target_samples - len(selected)
                n_select = min(len(county_indices), max_samples_per_county, remaining_needed)

                if n_select > 0:
                    county_selected = np.random.choice(county_indices, size=n_select, replace=False)
                    selected.extend(county_selected)
        else:
            # No per-county cap, just sample randomly
            selected = np.random.choice(pool, size=target_samples, replace=False)

        train_indices.extend(selected)
        source_breakdown["test_counties_historical"] = len(selected)

        # Track per-county distribution
        for idx in selected:
            fips = df.iloc[idx][fips_column]
            county_distribution[fips] = county_distribution.get(fips, 0) + 1

        logger.info(f"Added {len(selected)} samples from test counties (historical)")

    # Source 2: External counties
    external_config = data_sources.get("external_counties", {})
    if external_config.get("enabled", False):
        # Get all external county indices (not in test set)
        test_county_set = set(test_result.test_counties)
        all_external_mask = ~df[fips_column].isin(test_county_set)
        all_external_indices = df[all_external_mask].index.values

        # Sampling strategy for external counties
        strategy = external_config.get("sampling_strategy", "random")
        n_counties = external_config.get("n_counties", None)

        if strategy == "random":
            # Calculate target based on fraction of max_samples
            external_fraction = external_config.get("fraction", 1.0)

            if max_samples:
                # Calculate based on desired fraction of total
                target_external = int(max_samples * external_fraction)
                # Fill remaining budget if test counties had shortfall
                # This ensures we use the full max_samples budget when possible
                remaining_budget = max_samples - len(train_indices)
                target_external = max(target_external, remaining_budget)
            else:
                target_external = int(len(all_external_indices) * external_fraction)

            target_external = max(0, target_external)

            if target_external > 0 and len(all_external_indices) > 0:
                # Apply per-county cap if specified
                if max_samples_per_county:
                    # Group external indices by county
                    external_counties = {}
                    for idx in all_external_indices:
                        fips = df.iloc[idx][fips_column]
                        if fips not in external_counties:
                            external_counties[fips] = []
                        external_counties[fips].append(idx)

                    # Directly add samples from each county up to target
                    # Add all available samples (up to cap) until we hit target_external
                    selected = []
                    county_order = list(external_counties.keys())
                    np.random.shuffle(county_order)  # Random order for fairness

                    for fips in county_order:
                        if len(selected) >= target_external:
                            break

                        county_indices = external_counties[fips]
                        # Take min(county_size, cap, remaining_needed)
                        remaining_needed = target_external - len(selected)
                        n_select = min(len(county_indices), max_samples_per_county, remaining_needed)

                        if n_select > 0:
                            county_selected = np.random.choice(county_indices, size=n_select, replace=False)
                            selected.extend(county_selected)
                else:
                    # No per-county cap, just sample randomly
                    n_select = min(target_external, len(all_external_indices))
                    selected = np.random.choice(all_external_indices, size=n_select, replace=False)

                train_indices.extend(selected)
                source_breakdown["external_counties"] = len(selected)

                # Track per-county distribution
                for idx in selected:
                    fips = df.iloc[idx][fips_column]
                    county_distribution[fips] = county_distribution.get(fips, 0) + 1

                logger.info(f"Added {len(selected)} samples from external counties (random)")

        elif strategy == "stratified":
            # Stratified sample ensuring diverse county representation
            external_counties = df[all_external_mask][fips_column].unique()

            if n_counties:
                # Select subset of counties
                n_select_counties = min(n_counties, len(external_counties))
                selected_counties = np.random.choice(
                    external_counties, size=n_select_counties, replace=False
                )
            else:
                selected_counties = external_counties

            # Calculate target based on fraction
            external_fraction = external_config.get("fraction", 1.0)
            if max_samples:
                target_external = int(max_samples * external_fraction)
                # Fill remaining budget if test counties had shortfall
                # This ensures we use the full max_samples budget when possible
                remaining_budget = max_samples - len(train_indices)
                target_external = max(target_external, remaining_budget)
            else:
                target_external = external_config.get("samples_per_county", 100) * len(selected_counties)

            # Sample equally from each county
            if target_external > 0:
                samples_per_county = max(1, target_external // len(selected_counties))

                # Apply per-county cap if specified
                if max_samples_per_county:
                    samples_per_county = min(samples_per_county, max_samples_per_county)

                selected_indices = []
                for county in selected_counties:
                    county_mask = df[fips_column] == county
                    county_indices = df[county_mask].index.values
                    n_select = min(samples_per_county, len(county_indices))
                    if n_select > 0:
                        county_selected = np.random.choice(county_indices, size=n_select, replace=False)
                        selected_indices.extend(county_selected)
                        county_distribution[county] = len(county_selected)

                train_indices.extend(selected_indices)
                source_breakdown["external_counties"] = len(selected_indices)
                logger.info(f"Added {len(selected_indices)} samples from {len(selected_counties)} external counties (stratified)")

    train_indices = np.array(train_indices)

    # Final sampling if max_samples specified and we have too many
    max_samples = sampling_config.get("max_samples", None)
    if max_samples and len(train_indices) > max_samples:
        logger.info(f"Downsampling from {len(train_indices)} to {max_samples}")
        train_indices = np.random.choice(train_indices, size=max_samples, replace=False)

        # Recalculate county distribution after downsampling
        county_distribution = {}
        for idx in train_indices:
            fips = df.iloc[idx][fips_column]
            county_distribution[fips] = county_distribution.get(fips, 0) + 1

    metadata = {
        "version": config.get("version", "unknown"),
        "description": config.get("description", ""),
        "n_train_samples": len(train_indices),
        "n_counties_used": len(county_distribution),
        "random_seed": random_seed,
    }

    logger.info(f"Train set created: {len(train_indices)} samples from {len(county_distribution)} counties")

    return TrainSetResult(
        train_indices=train_indices,
        source_breakdown=source_breakdown,
        county_distribution=county_distribution,
        metadata=metadata
    )


# ==============================================================================
# CONFIG LOADING
# ==============================================================================

def load_test_set_config(config_path: str) -> Dict:
    """Load test set configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_train_set_config(config_path: str) -> Dict:
    """Load train set configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def create_test_train_split(
    df: pd.DataFrame,
    test_config: Dict,
    train_config: Dict,
    fips_column: str = "fips",
    date_column: str = "sale_date",
    random_seed: int = 42
) -> Tuple[TestSetResult, TrainSetResult]:
    """
    Create both test and train sets in one call.

    Args:
        df: Full preprocessed DataFrame
        test_config: Test set configuration
        train_config: Train set configuration
        fips_column: Column name for county FIPS codes
        date_column: Column name for sale dates
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (TestSetResult, TrainSetResult)
    """
    test_result = create_test_set(
        df=df,
        config=test_config,
        fips_column=fips_column,
        date_column=date_column,
        random_seed=random_seed
    )

    train_result = create_train_set(
        df=df,
        config=train_config,
        test_result=test_result,
        fips_column=fips_column,
        random_seed=random_seed
    )

    return test_result, train_result


def get_train_test_data(
    df: pd.DataFrame,
    test_result: TestSetResult,
    train_result: TrainSetResult,
    target_column: str = "SALE_AMOUNT",
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extract actual train/test DataFrames from split results.

    Args:
        df: Full preprocessed DataFrame
        test_result: TestSetResult from create_test_set
        train_result: TrainSetResult from create_train_set
        target_column: Name of target column
        exclude_columns: Columns to exclude from features (e.g., ['fips', 'CLIP'])

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Default exclude columns include administrative columns that shouldn't be features
    # These should have been dropped in Phase 1 preprocessing but may still be present
    # in older datasets or due to preprocessing bugs
    if exclude_columns is None:
        exclude_columns = [
            # ID/administrative columns
            "fips", "CLIP", "sale_date",
            # Additional administrative columns that may not have been dropped in Phase 1
            "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
            "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
            "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
            "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
            "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
            # Baseline value - kept for evaluation but excluded from training features
            "CALCULATED_TOTAL_VALUE",
        ]

    # Get train data
    train_df = df.iloc[train_result.train_indices]
    y_train = train_df[target_column]
    X_train = train_df.drop(columns=[target_column] + [c for c in exclude_columns if c in train_df.columns])

    # Get test data
    test_df = df.iloc[test_result.test_indices]
    y_test = test_df[target_column]
    X_test = test_df.drop(columns=[target_column] + [c for c in exclude_columns if c in test_df.columns])

    # Additional safety check: drop any remaining object dtype columns
    # This handles edge cases where new non-numeric columns were added
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        logger.warning(f"Dropping {len(object_cols)} unexpected object columns: {object_cols}")
        X_train = X_train.drop(columns=object_cols)
        X_test = X_test.drop(columns=[c for c in object_cols if c in X_test.columns])

    return X_train, y_train, X_test, y_test


# ==============================================================================
# SAVE/LOAD FUNCTIONS
# ==============================================================================

def save_test_set_result(result: TestSetResult, output_dir: str, df: pd.DataFrame = None,
                          target_column: str = "SALE_AMOUNT", baseline_column: str = "CALCULATED_TOTAL_VALUE"):
    """
    Save test set result to disk.

    Args:
        result: TestSetResult to save
        output_dir: Directory to save files to
        df: Optional full DataFrame for extracting baseline values
        target_column: Column name for target variable
        baseline_column: Column name for baseline predictions

    Creates:
        output_dir/
        ├── test_indices.npy
        ├── train_pool_indices.npy
        ├── test_baseline_values.npy      # NEW: baseline values for test set
        ├── test_sale_amounts.npy         # NEW: sale amounts for test set
        ├── train_pool_baseline_values.npy # NEW: baseline values for train pool
        ├── train_pool_sale_amounts.npy    # NEW: sale amounts for train pool
        ├── test_counties.json
        ├── county_info.json
        ├── size_buckets.json
        └── metadata.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(output_path / "test_indices.npy", result.test_indices)
    np.save(output_path / "train_pool_indices.npy", result.train_pool_indices)

    # Save baseline values if df is provided
    if df is not None:
        if baseline_column in df.columns:
            test_baseline = df.iloc[result.test_indices][baseline_column].values
            np.save(output_path / "test_baseline_values.npy", test_baseline)
            logger.info(f"Saved test baseline values ({len(test_baseline)} samples)")

            train_pool_baseline = df.iloc[result.train_pool_indices][baseline_column].values
            np.save(output_path / "train_pool_baseline_values.npy", train_pool_baseline)
            logger.info(f"Saved train pool baseline values ({len(train_pool_baseline)} samples)")
        else:
            logger.warning(f"Column '{baseline_column}' not found in DataFrame, skipping baseline value extraction")

        if target_column in df.columns:
            test_sales = df.iloc[result.test_indices][target_column].values
            np.save(output_path / "test_sale_amounts.npy", test_sales)
            logger.info(f"Saved test sale amounts ({len(test_sales)} samples)")

            train_pool_sales = df.iloc[result.train_pool_indices][target_column].values
            np.save(output_path / "train_pool_sale_amounts.npy", train_pool_sales)
            logger.info(f"Saved train pool sale amounts ({len(train_pool_sales)} samples)")
        else:
            logger.warning(f"Column '{target_column}' not found in DataFrame, skipping sale amount extraction")

    # Save JSON files
    import json

    # Convert test_counties to regular Python ints
    test_counties_list = [int(x) for x in result.test_counties]
    with open(output_path / "test_counties.json", 'w') as f:
        json.dump(test_counties_list, f, indent=2)

    # Convert county_info keys to strings and ensure all numeric values are Python types
    county_info_str = {}
    for k, v in result.county_info.items():
        county_info_str[str(k)] = {
            key: int(val) if isinstance(val, (np.integer, np.int64, np.int32)) else float(val) if isinstance(val, (np.floating, np.float64, np.float32)) else val
            for key, val in v.items()
        }
    with open(output_path / "county_info.json", 'w') as f:
        json.dump(county_info_str, f, indent=2)

    # Convert size_buckets values to regular Python ints
    size_buckets_converted = {
        k: [int(x) for x in v] for k, v in result.size_buckets.items()
    }
    with open(output_path / "size_buckets.json", 'w') as f:
        json.dump(size_buckets_converted, f, indent=2)

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(result.metadata, f, indent=2)

    logger.info(f"Saved test set to {output_dir}")


def load_test_set_result(split_dir: str) -> TestSetResult:
    """
    Load pre-generated test set from disk.

    Args:
        split_dir: Directory containing saved test set

    Returns:
        TestSetResult
    """
    split_path = Path(split_dir)

    # Load numpy arrays
    test_indices = np.load(split_path / "test_indices.npy")
    train_pool_indices = np.load(split_path / "train_pool_indices.npy")

    # Load JSON files
    import json

    with open(split_path / "test_counties.json", 'r') as f:
        test_counties = json.load(f)

    with open(split_path / "county_info.json", 'r') as f:
        county_info_str = json.load(f)
        # Convert keys back to integers
        county_info = {int(k): v for k, v in county_info_str.items()}

    with open(split_path / "size_buckets.json", 'r') as f:
        size_buckets = json.load(f)

    with open(split_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded test set from {split_dir}")
    logger.info(f"  Test counties: {len(test_counties)}")
    logger.info(f"  Test samples: {len(test_indices)}")
    logger.info(f"  Train pool samples: {len(train_pool_indices)}")

    return TestSetResult(
        test_counties=test_counties,
        test_indices=test_indices,
        train_pool_indices=train_pool_indices,
        county_info=county_info,
        size_buckets=size_buckets,
        metadata=metadata
    )


def save_train_set_result(result: TrainSetResult, output_dir: str, df: pd.DataFrame = None,
                          target_column: str = "SALE_AMOUNT", baseline_column: str = "CALCULATED_TOTAL_VALUE",
                          log_transformed_target: bool = True):
    """
    Save train set result to disk.

    Args:
        result: TrainSetResult to save
        output_dir: Directory to save files to
        df: Optional full DataFrame for extracting baseline values
        target_column: Column name for target variable
        baseline_column: Column name for baseline predictions
        log_transformed_target: Whether the target variable is log-transformed (default: True)

    Creates:
        output_dir/
        ├── train_indices.npy
        ├── train_baseline_values.npy     # NEW: baseline values for training set
        ├── train_sale_amounts.npy        # NEW: sale amounts for training set
        ├── source_breakdown.json
        ├── county_distribution.json
        └── metadata.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save numpy array
    np.save(output_path / "train_indices.npy", result.train_indices)

    # Save baseline values if df is provided
    if df is not None:
        if baseline_column in df.columns:
            train_baseline = df.iloc[result.train_indices][baseline_column].values
            np.save(output_path / "train_baseline_values.npy", train_baseline)

            # Compute and save adjustment ratio (median of sale_amount / baseline)
            train_sales = df.iloc[result.train_indices][target_column].values

            # If target is log-transformed, exp-transform it to get original scale
            if log_transformed_target:
                train_sales_original = np.exp(train_sales)
            else:
                train_sales_original = train_sales

            # Filter out zeros and invalid values before computing ratio
            valid_mask = (train_baseline > 0) & (train_sales_original > 0) & np.isfinite(train_baseline) & np.isfinite(train_sales_original)
            if valid_mask.sum() > 0:
                ratios = train_sales_original[valid_mask] / train_baseline[valid_mask]
                adjustment_ratio = float(np.median(ratios))
            else:
                adjustment_ratio = 1.0
                logger.warning("No valid baseline/sales pairs found, using adjustment ratio of 1.0")

            # Add adjustment ratio to metadata
            result.metadata['baseline_adjustment_ratio'] = adjustment_ratio
            logger.info(f"Computed baseline adjustment ratio: {adjustment_ratio:.4f}")
            logger.info(f"Saved train baseline values ({len(train_baseline)} samples)")
        else:
            logger.warning(f"Column '{baseline_column}' not found in DataFrame, skipping baseline value extraction")

        if target_column in df.columns:
            train_sales = df.iloc[result.train_indices][target_column].values
            np.save(output_path / "train_sale_amounts.npy", train_sales)
            logger.info(f"Saved train sale amounts ({len(train_sales)} samples)")
        else:
            logger.warning(f"Column '{target_column}' not found in DataFrame, skipping sale amount extraction")

    # Save JSON files
    import json

    # Convert source_breakdown values to Python ints
    source_breakdown_converted = {
        k: int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v
        for k, v in result.source_breakdown.items()
    }
    with open(output_path / "source_breakdown.json", 'w') as f:
        json.dump(source_breakdown_converted, f, indent=2)

    # Convert county_distribution keys to strings and values to Python ints
    county_dist_str = {
        str(k): int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v
        for k, v in result.county_distribution.items()
    }
    with open(output_path / "county_distribution.json", 'w') as f:
        json.dump(county_dist_str, f, indent=2)

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(result.metadata, f, indent=2)

    logger.info(f"Saved train set to {output_dir}")


def load_train_set_result(split_dir: str) -> TrainSetResult:
    """
    Load pre-generated train set from disk.

    Args:
        split_dir: Directory containing saved train set

    Returns:
        TrainSetResult
    """
    split_path = Path(split_dir)

    # Load numpy array
    train_indices = np.load(split_path / "train_indices.npy")

    # Load JSON files
    import json

    with open(split_path / "source_breakdown.json", 'r') as f:
        source_breakdown = json.load(f)

    with open(split_path / "county_distribution.json", 'r') as f:
        county_dist_str = json.load(f)
        # Convert keys back to integers
        county_distribution = {int(k): v for k, v in county_dist_str.items()}

    with open(split_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded train set from {split_dir}")
    logger.info(f"  Train samples: {len(train_indices)}")
    logger.info(f"  Counties used: {len(county_distribution)}")
    logger.info(f"  Source breakdown: {source_breakdown}")

    return TrainSetResult(
        train_indices=train_indices,
        source_breakdown=source_breakdown,
        county_distribution=county_distribution,
        metadata=metadata
    )
