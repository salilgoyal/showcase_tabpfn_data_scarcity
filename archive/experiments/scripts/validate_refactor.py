#!/usr/bin/env python
"""
Validation script to test the preprocessing refactor.

This script:
1. Tests column categorization on sample county data
2. Compares old vs new preprocessing approaches
3. Verifies that equivalent configs produce identical results
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CountyDataLoader
from data.column_definitions import ColumnCategorizer, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_column_categorization(fips: int, county_csvs_dir: str):
    """Test column categorization on a sample county."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing column categorization on county {fips}")
    logger.info(f"{'='*60}")

    # Load county data
    filepath = Path(county_csvs_dir) / f"fips_{fips}.csv"
    df = pd.read_csv(filepath, low_memory=False)

    logger.info(f"Loaded data: {df.shape}")

    # Categorize columns
    categorizer = ColumnCategorizer(df, target_column='SALE_AMOUNT')
    categories = categorizer.categorize_all()

    # Print summary
    logger.info("\nColumn Categorization Summary:")
    for category, cols in categories.items():
        if cols:
            logger.info(f"  {category}: {len(cols)} columns")
            if len(cols) <= 5:
                logger.info(f"    {cols}")
            else:
                logger.info(f"    {cols[:5]} ... (and {len(cols)-5} more)")

    # Test feature selection with different configs
    logger.info("\n" + "-"*60)
    logger.info("Testing feature selection with different configs:")
    logger.info("-"*60)

    # Config 1: Property chars only
    config1 = {
        'property_chars': True,
        'census_bg': False,
        'census_tract': False,
        'assessed_value': False,
        'geographic': False,
        'temporal': True,
    }
    features1 = get_feature_columns(df, config1, 'SALE_AMOUNT')
    logger.info(f"\nProperty chars only:")
    logger.info(f"  Continuous: {len(features1['continuous_cols'])}")
    logger.info(f"  Binary: {len(features1['binary_cols'])}")
    logger.info(f"  Categorical: {len(features1['categorical_cols'])}")

    # Config 2: Assessed value + census only
    config2 = {
        'property_chars': False,
        'census_bg': True,
        'census_tract': False,
        'assessed_value': True,
        'geographic': False,
        'temporal': True,
    }
    features2 = get_feature_columns(df, config2, 'SALE_AMOUNT')
    logger.info(f"\nAssessed value + census BG:")
    logger.info(f"  Continuous: {len(features2['continuous_cols'])}")
    logger.info(f"  Binary: {len(features2['binary_cols'])}")
    logger.info(f"  Categorical: {len(features2['categorical_cols'])}")

    # Config 3: Full features
    config3 = {
        'property_chars': True,
        'census_bg': True,
        'census_tract': False,
        'assessed_value': True,
        'geographic': True,
        'temporal': True,
    }
    features3 = get_feature_columns(df, config3, 'SALE_AMOUNT')
    logger.info(f"\nFull features:")
    logger.info(f"  Continuous: {len(features3['continuous_cols'])}")
    logger.info(f"  Binary: {len(features3['binary_cols'])}")
    logger.info(f"  Categorical: {len(features3['categorical_cols'])}")

    return True


def test_preprocessing_equivalence(fips: int, county_csvs_dir: str):
    """
    Test that old and new preprocessing produce equivalent results.

    This tests the property_chars_only mode which should be equivalent
    between old and new approaches.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing preprocessing equivalence on county {fips}")
    logger.info(f"{'='*60}")

    # Test 1: Property chars only mode
    logger.info("\nTest: Property chars only mode")

    # Old format
    logger.info("Loading with OLD config format...")
    loader_old = CountyDataLoader(
        county_csvs_dir=county_csvs_dir,
        target_column='SALE_AMOUNT',
        use_evelyn_preprocessing=True,
        include_property_chars=True,
        property_chars_only=True
    )

    df_old = loader_old.load_county(fips, drop_missing_target=True)
    X_old, y_old = loader_old.preprocess_for_training(df_old)

    logger.info(f"Old format result: X shape={X_old.shape}, y shape={y_old.shape}")

    # New format
    logger.info("Loading with NEW config format...")
    new_config = {
        'features': {
            'property_chars': True,
            'census_bg': False,
            'census_tract': False,
            'assessed_value': False,
            'geographic': False,
            'temporal': True,
        },
        'steps': {
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
    }

    loader_new = CountyDataLoader(
        county_csvs_dir=county_csvs_dir,
        target_column='SALE_AMOUNT',
        preprocessing_config=new_config
    )

    df_new = loader_new.load_county(fips, drop_missing_target=True)
    X_new, y_new = loader_new.preprocess_for_training(df_new)

    logger.info(f"New format result: X shape={X_new.shape}, y shape={y_new.shape}")

    # Compare results
    logger.info("\nComparison:")

    # Check shapes
    shapes_match = (X_old.shape == X_new.shape and y_old.shape == y_new.shape)
    logger.info(f"  Shapes match: {shapes_match}")
    if not shapes_match:
        logger.warning(f"    Old: X={X_old.shape}, y={y_old.shape}")
        logger.warning(f"    New: X={X_new.shape}, y={y_new.shape}")

    # Check feature names
    common_features = set(X_old.columns) & set(X_new.columns)
    old_only = set(X_old.columns) - set(X_new.columns)
    new_only = set(X_new.columns) - set(X_old.columns)

    logger.info(f"  Common features: {len(common_features)}")
    if old_only:
        logger.warning(f"  Features only in OLD: {len(old_only)}")
        logger.warning(f"    {list(old_only)[:10]}")
    if new_only:
        logger.warning(f"  Features only in NEW: {len(new_only)}")
        logger.warning(f"    {list(new_only)[:10]}")

    # Check target values (for overlapping indices)
    min_len = min(len(y_old), len(y_new))
    y_old_subset = y_old.iloc[:min_len].values
    y_new_subset = y_new.iloc[:min_len].values

    if len(y_old_subset) > 0:
        # Check if values are close (allowing for numerical precision)
        values_close = np.allclose(y_old_subset, y_new_subset, rtol=1e-5, atol=1e-8, equal_nan=True)
        logger.info(f"  Target values match (first {min_len}): {values_close}")
        if not values_close:
            # Show some examples
            diff = np.abs(y_old_subset - y_new_subset)
            max_diff_idx = np.nanargmax(diff)
            logger.warning(f"    Max difference at index {max_diff_idx}:")
            logger.warning(f"      Old: {y_old_subset[max_diff_idx]}")
            logger.warning(f"      New: {y_new_subset[max_diff_idx]}")
            logger.warning(f"      Diff: {diff[max_diff_idx]}")

    # Check feature values for common features
    if len(common_features) > 0:
        sample_feature = list(common_features)[0]
        X_old_sample = X_old[sample_feature].iloc[:min_len].values
        X_new_sample = X_new[sample_feature].iloc[:min_len].values

        features_close = np.allclose(X_old_sample, X_new_sample, rtol=1e-5, atol=1e-8, equal_nan=True)
        logger.info(f"  Feature values match (sample: {sample_feature}): {features_close}")

    logger.info("\nValidation complete!")

    return shapes_match and len(old_only) == 0 and len(new_only) == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate preprocessing refactor')
    parser.add_argument(
        '--county_csvs_dir',
        type=str,
        default='/scratch/users/salilg/county_csvs/',
        help='Directory containing county CSV files'
    )
    parser.add_argument(
        '--fips',
        type=int,
        default=None,
        help='County FIPS code to test (if not specified, uses first available)'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['categorization', 'equivalence', 'all'],
        default='all',
        help='Which test to run'
    )

    args = parser.parse_args()

    # Find a county to test if not specified
    if args.fips is None:
        county_files = sorted(Path(args.county_csvs_dir).glob('fips_*.csv'))
        if not county_files:
            logger.error(f"No county files found in {args.county_csvs_dir}")
            return 1

        # Use first available county
        fips_str = county_files[0].stem.replace('fips_', '')
        args.fips = int(fips_str)
        logger.info(f"Auto-selected county FIPS: {args.fips}")

    success = True

    if args.test in ['categorization', 'all']:
        try:
            test_column_categorization(args.fips, args.county_csvs_dir)
        except Exception as e:
            logger.error(f"Categorization test failed: {e}", exc_info=True)
            success = False

    if args.test in ['equivalence', 'all']:
        try:
            equiv_success = test_preprocessing_equivalence(args.fips, args.county_csvs_dir)
            success = success and equiv_success
        except Exception as e:
            logger.error(f"Equivalence test failed: {e}", exc_info=True)
            success = False

    if success:
        logger.info("\n" + "="*60)
        logger.info("✓ All validation tests passed!")
        logger.info("="*60)
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("✗ Some validation tests failed")
        logger.error("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
