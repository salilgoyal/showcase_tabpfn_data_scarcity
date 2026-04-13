"""
Wrapper for Evelyn's preprocessing pipeline to integrate with existing experiment code.

This module provides a drop-in replacement for load_and_prepare_data() that uses
Evelyn's comprehensive preprocessing (winsorization, log transformation, normalization, etc.)
while maintaining the same interface as the original data_utils.py functions.

NEW: Supports fine-grained control over feature selection and preprocessing steps via config.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add evelyn_files to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'evelyn_files'))
from preprocess import Preprocess

# Import new column categorization module
from .column_definitions import get_feature_columns

logger = logging.getLogger(__name__)


def get_evelyn_column_types(df, label_col='SALE_AMOUNT', include_property_chars=False, property_chars_only=False):
    """
    Get column types matching Evelyn's exact feature selection from census_loop_config.yaml.

    Args:
        df: DataFrame to analyze
        label_col: Label column to exclude
        include_property_chars: If True, add property characteristics to continuous features
        property_chars_only: If True, drop assessed value, census, and other non-property columns

    Returns:
        continuous_cols, binary_cols, categorical_cols, meta_cols
    """
    # Drop columns if property_chars_only mode
    if property_chars_only:
        cols_to_drop = [
            'Unnamed: 0',
            'ASSESSED_YEAR',
            'CENSUS_ID',
            'MARKET_TOTAL_VALUE',
            'latitude',
            'longitude',
            'address',
            'TOTAL_TAX_AMOUNT',
            'NET_TAX_AMOUNT',
            'TAX_RATE_AREA_CODE',
            'CLIP',
            'SALE_AMOUNT',
            'PREVIOUS_CLIP',
            'OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID',
            'sale_date',
            'ASSESSED_TOTAL_VALUE',
            'APPRAISED_TOTAL_VALUE',
            'CALCULATED_TOTAL_VALUE',
            'CALCULATED_TOTAL_VALUE_SOURCE_CODE',
            'tract',
            'block_group',
            'census_pct_children_tract',
            'census_pct_senior_tract',
            'census_med_age_tract',
            'census_pct_married_hh_tract',
            'census_pct_single_hh_tract',
            'census_pct_high_school_tract',
            'census_pct_college_tract',
            'census_pct_graduate_tract',
            'census_pct_poverty_tract',
            'census_med_hh_inc_tract',
            'census_med_per_cap_inc_tract',
            'census_pct_snap_tract',
            'census_unemp_rate_tract',
            'census_med_yr_built_tract',
            'census_pct_renter_occ_tract',
            'census_med_rent_tract',
            'tract_id',
            'census_pct_children_bg',
            'census_pct_senior_bg',
            'census_med_age_bg',
            'census_pct_married_hh_bg',
            'census_pct_single_hh_bg',
            'census_pct_high_school_bg',
            'census_pct_college_bg',
            'census_pct_graduate_bg',
            'census_pct_poverty_bg',
            'census_med_hh_inc_bg',
            'census_med_per_cap_inc_bg',
            'census_pct_snap_bg',
            'census_unemp_rate_bg',
            'census_med_yr_built_bg',
            'census_pct_renter_occ_bg',
            'census_med_rent_bg',
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        logger.info(f"Property chars only mode: dropped {len([col for col in cols_to_drop if col in df.columns])} columns")


    # ====================================================================
    # FEATURE SET SELECTION
    # ====================================================================

    if property_chars_only:
        # Property chars only: start with empty continuous_cols
        # (char_* features will be added below)
        continuous_cols = []
    else:
        # Standard mode: include assessed value + census features
        continuous_cols = ['CALCULATED_TOTAL_VALUE']

        census_cols = [
            'census_pct_children_bg',
            'census_pct_senior_bg',
            'census_med_age_bg',
            'census_pct_married_hh_bg',
            'census_pct_single_hh_bg',
            'census_pct_high_school_bg',
            'census_pct_college_bg',
            'census_pct_graduate_bg',
            'census_pct_poverty_bg',
            'census_med_hh_inc_bg',
            'census_med_per_cap_inc_bg',
            'census_pct_snap_bg',
            'census_unemp_rate_bg',
            'census_med_yr_built_bg',
            'census_pct_renter_occ_bg'
        ]

        # Only add census columns that exist in the data
        for col in census_cols:
            if col in df.columns:
                continuous_cols.append(col)

    # ====================================================================
    # PROPERTY CHARACTERISTICS
    # ====================================================================
    if include_property_chars or property_chars_only:
        # Add all char_* columns (property characteristics)
        char_cols = [col for col in df.columns if col.startswith('char_')]
        # Note: latitude/longitude already dropped in property_chars_only mode
        if include_property_chars and not property_chars_only:
            if 'latitude' in df.columns:
                char_cols.append('latitude')
            if 'longitude' in df.columns:
                char_cols.append('longitude')
        continuous_cols.extend(char_cols)
        logger.info(f"Including {len(char_cols)} property characteristic features")

    # Categorical and binary features
    # Evelyn's config has empty lists, but we'll detect categoricals if property chars are included
    if include_property_chars or property_chars_only:
        # Binary features: missing indicators
        binary_cols = [col for col in df.columns if col.endswith('_miss') and col not in continuous_cols]
        # Categorical: low-cardinality columns
        categorical_cols = [col for col in df.columns
                          if col.endswith('_cat') and col not in continuous_cols and col != label_col]
    else:
        categorical_cols = []
        binary_cols = []

    # Meta columns (everything else that's not the label or sale_date)
    # These are kept for record-keeping but NOT used in modeling
    meta_cols = [
        'meta_sfh', 'ASSESSED_YEAR', 'SALE_YEAR', 'CLIP', 'fips',
        'tract', 'block_group', 'MARKET_TOTAL_VALUE', 'ASSESSED_TOTAL_VALUE',
        'APPRAISED_TOTAL_VALUE', 'CALCULATED_TOTAL_VALUE_SOURCE_CODE',
        'TOTAL_TAX_AMOUNT', 'NET_TAX_AMOUNT', 'latitude', 'longitude',
        'address', 'PREVIOUS_CLIP', 'OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID',
        'block_group_id', 'tract_id', 'CENSUS_ID', 'TAX_RATE_AREA_CODE',
        'MULTI_OR_SPLIT_PARCEL_CODE'
    ]

    # Only keep meta columns that exist in data
    meta_cols = [col for col in meta_cols if col in df.columns and col != label_col]

    if property_chars_only:
        logger.info(f"Feature set (PROPERTY CHARS ONLY): {len(continuous_cols)} continuous + {len(binary_cols)} binary + {len(categorical_cols)} categorical")
        logger.info(f"Includes: property characteristics (char_*) ONLY + time vars (generated)")
        logger.info(f"Excludes: CALCULATED_TOTAL_VALUE, census_*, lat/lon")
    elif include_property_chars:
        logger.info(f"Feature set: {len(continuous_cols)} continuous + {len(binary_cols)} binary + {len(categorical_cols)} categorical")
        logger.info(f"Includes: CALCULATED_TOTAL_VALUE + census vars + property characteristics + time vars")
    else:
        logger.info(f"Feature set (Evelyn's minimal): {len(continuous_cols)} continuous features")
        logger.info(f"Includes: CALCULATED_TOTAL_VALUE + {len(continuous_cols)-1} census vars + time vars (generated)")

    return continuous_cols, binary_cols, categorical_cols, meta_cols, df


def load_and_prepare_data_evelyn(data_path, cbg_column='block_group_id',
                                  skip_mice=True, wins_pctile=1, log_label=True,
                                  include_property_chars=False, property_chars_only=False):
    """
    Load and preprocess Cook County data using Evelyn's preprocessing pipeline.

    This is a drop-in replacement for data_utils.load_and_prepare_data() that applies:
    - Winsorization of outliers
    - Log transformation of target
    - Normalization of features
    - One-hot encoding of categoricals
    - Train-test temporal split (last year as test)

    Args:
        data_path: Path to cook_county.csv
        cbg_column: Name of CBG column (ignored - not used in this preprocessing)
        skip_mice: If True, skip MICE imputation (faster, for experiments)
        wins_pctile: Percentile for winsorization (default: 1)
        log_label: Whether to apply log transformation to target (default: True)
        include_property_chars: If True, include property characteristics (char_* features)
                                If False, use only CALCULATED_TOTAL_VALUE + census + time
                                (default: False, matching Evelyn's config)
        property_chars_only: If True, use ONLY property characteristics (drops assessed value & census)

    Returns:
        X_train_pool, y_train_pool, X_test, y_test, cbg_column_name

    Note: y_train_pool and y_test are LOG-TRANSFORMED if log_label=True.
          Use calculate_metrics with log_transformed=True to auto-inverse-transform.
    """
    logger.info(f"Loading data from {data_path} with Evelyn's preprocessing")
    logger.info(f"include_property_chars={include_property_chars}, property_chars_only={property_chars_only}")

    # Load raw data
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded data shape: {df.shape}")

    # Get column types - with or without property characteristics
    continuous_cols, binary_cols, categorical_cols, meta_cols, df = get_evelyn_column_types(
        df,
        label_col='SALE_AMOUNT',
        include_property_chars=include_property_chars,
        property_chars_only=property_chars_only
    )

    # Initialize Evelyn's Preprocess class
    preprocessor = Preprocess(
        data=df,
        label='SALE_AMOUNT',
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        meta_cols=meta_cols,
        sale_date_col='sale_date',
        geography=None,  # Don't use geography column
        share_non_null=0.5,  # Match Evelyn's config (requires 50% non-null to keep column)
        random_state=42,
        wins_pctile=wins_pctile,
        log_label=log_label,
        mice_iters=3,
        test_size=0.2,
        log_dir='logs'
    )

    # Run preprocessing pipeline (but skip MICE and normalization for now)
    logger.info("Running Evelyn's preprocessing pipeline...")

    # We'll manually control the pipeline to ensure proper ordering
    preprocessor.drop_null_labels()
    preprocessor.gen_time_vars()
    preprocessor.drop_single_value_cols()
    preprocessor.drop_mostly_null_cols()
    preprocessor.drop_lowest_ratios()
    preprocessor.drop_repeat_sales()
    preprocessor.one_hot()
    preprocessor.renumber_geo_col()

    # Train-test split (temporal)
    preprocessor.train_test_split()

    # ====================================================================
    # FIX STRING COLUMNS BEFORE WINSORIZATION
    # Some continuous columns may have been incorrectly classified or contain strings
    # ====================================================================
    for col in preprocessor._continuous_cols:
        if col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].dtype == 'object':
                logger.warning(f"Column {col} is object type in continuous_cols, attempting to convert")
                try:
                    preprocessor.X_train[col] = pd.to_numeric(preprocessor.X_train[col], errors='coerce')
                    preprocessor.X_test[col] = pd.to_numeric(preprocessor.X_test[col], errors='coerce')
                except:
                    logger.warning(f"Failed to convert {col}, removing from continuous_cols")
                    preprocessor._continuous_cols.remove(col)

    # Winsorize continuous features and labels
    if wins_pctile > 0:
        preprocessor.winsorize_continuous()
        preprocessor.winsorize_label()

    # Log transform labels
    preprocessor.log_label()

    # Drop problematic columns after split
    preprocessor._drop_problematic_cols_from_splits()

    # Skip MICE and use simple imputation instead (much faster)
    if skip_mice:
        logger.info("Skipping MICE imputation, using median/mode imputation")
        for col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].isnull().any():
                if pd.api.types.is_numeric_dtype(preprocessor.X_train[col]):
                    fill_value = preprocessor.X_train[col].median()
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
                else:
                    fill_value = preprocessor.X_train[col].mode()[0] if len(preprocessor.X_train[col].mode()) > 0 else 0
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
    else:
        preprocessor.impute_missings_with_mice()

    # Normalize continuous columns
    preprocessor.normalize_continuous_cols()

    X_train = preprocessor.X_train
    y_train = preprocessor.y_train
    X_test = preprocessor.X_test
    y_test = preprocessor.y_test

    # ====================================================================
    # FIX INDEX MISMATCH: Reset index so sample_train_set can work
    # Evelyn's preprocessing drops rows, so indices are no longer sequential
    # ====================================================================
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ====================================================================
    # FIX STRING COLUMNS: Convert all columns to numeric
    # Some columns may be strings after preprocessing which breaks winsorization
    # ====================================================================
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                logger.warning(f"Converted column {col} from object to numeric")
            except:
                logger.warning(f"Could not convert column {col} to numeric, dropping it")
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    logger.info(f"Preprocessing complete!")
    logger.info(f"Train pool shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Target log-transformed: {log_label}")

    # Return in same format as original load_and_prepare_data
    # Note: X_train is the full training set (can be subsampled later)
    # Note: y_train and y_test are log-transformed if log_label=True
    return X_train, y_train, X_test, y_test, cbg_column


# ============================================================================
# NEW MODULAR PREPROCESSING FUNCTION
# ============================================================================

def load_and_prepare_data(
    data_path: str,
    feature_config: dict,
    step_config: dict,
    cbg_column: str = 'block_group_id'
):
    """
    Load and preprocess data with fine-grained control over features and steps.

    This is the NEW recommended function that uses the modular column_definitions
    system for feature selection and conditional preprocessing step execution.

    Args:
        data_path: Path to CSV file
        feature_config: Feature selection flags:
            {
                'property_chars': bool,
                'census_bg': bool,
                'census_tract': bool,
                'assessed_value': bool,
                'geographic': bool,
                'temporal': bool,
            }
        step_config: Preprocessing step flags:
            {
                'drop_null_labels': bool,
                'drop_single_value_cols': bool,
                'drop_mostly_null_cols': bool,
                'share_non_null': float,
                'drop_lowest_ratios': bool,
                'drop_repeat_sales': bool,
                'generate_temporal_features': bool,
                'one_hot_encode': bool,
                'winsorize': bool,
                'winsorize_percentile': int,
                'log_transform_target': bool,
                'normalize_continuous': bool,
                'normalize_binary': bool,
                'impute_method': str,  # "median", "mean", "mice", "none"
                'mice_iterations': int,
            }
        cbg_column: Name of CBG column (for interface compatibility)

    Returns:
        X_train, y_train, X_test, y_test, cbg_column

    Note:
        - y_train and y_test are LOG-TRANSFORMED if step_config['log_transform_target']=True
        - Make sure to pass log_transformed=True to compute_metrics() in that case
    """
    logger.info(f"Loading data from {data_path} with modular preprocessing")
    logger.info(f"Feature config: {feature_config}")
    logger.info(f"Step config keys: {list(step_config.keys())}")

    # Load raw data
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded data shape: {df.shape}")

    # Get feature columns using new column categorization system
    feature_cols = get_feature_columns(df, feature_config, target_column='SALE_AMOUNT')

    continuous_cols = feature_cols['continuous_cols']
    binary_cols = feature_cols['binary_cols']
    categorical_cols = feature_cols['categorical_cols']
    meta_cols = feature_cols['meta_cols']

    logger.info(f"Selected features: {len(continuous_cols)} continuous, "
                f"{len(binary_cols)} binary, {len(categorical_cols)} categorical")

    # Initialize Evelyn's Preprocess class
    preprocessor = Preprocess(
        data=df,
        label='SALE_AMOUNT',
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        meta_cols=meta_cols,
        sale_date_col='sale_date',
        geography=None,
        share_non_null=step_config.get('share_non_null', 0.5),
        random_state=42,
        wins_pctile=step_config.get('winsorize_percentile', 1),
        log_label=step_config.get('log_transform_target', True),
        mice_iters=step_config.get('mice_iterations', 3),
        test_size=0.2,
        log_dir='logs'
    )

    # ========================================================================
    # CONDITIONAL PREPROCESSING PIPELINE
    # Apply each step based on step_config flags
    # ========================================================================

    logger.info("Running conditional preprocessing pipeline...")

    # Data cleaning
    if step_config.get('drop_null_labels', True):
        logger.debug("  - Dropping null labels")
        preprocessor.drop_null_labels()

    if step_config.get('generate_temporal_features', True):
        logger.debug("  - Generating temporal features")
        preprocessor.gen_time_vars()

    if step_config.get('drop_single_value_cols', True):
        logger.debug("  - Dropping single-value columns")
        preprocessor.drop_single_value_cols()

    if step_config.get('drop_mostly_null_cols', True):
        logger.debug("  - Dropping mostly-null columns")
        preprocessor.drop_mostly_null_cols()

    if step_config.get('drop_lowest_ratios', True):
        logger.debug("  - Dropping lowest ratio columns")
        preprocessor.drop_lowest_ratios()

    if step_config.get('drop_repeat_sales', True):
        logger.debug("  - Dropping repeat sales")
        preprocessor.drop_repeat_sales()

    # Feature engineering
    if step_config.get('one_hot_encode', True):
        logger.debug("  - One-hot encoding categoricals")
        preprocessor.one_hot()

    preprocessor.renumber_geo_col()

    # Train-test split (always required)
    logger.debug("  - Train-test split")
    preprocessor.train_test_split()

    # Fix string columns before winsorization
    for col in preprocessor._continuous_cols:
        if col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].dtype == 'object':
                logger.warning(f"Column {col} is object type, converting to numeric")
                try:
                    preprocessor.X_train[col] = pd.to_numeric(
                        preprocessor.X_train[col], errors='coerce'
                    )
                    preprocessor.X_test[col] = pd.to_numeric(
                        preprocessor.X_test[col], errors='coerce'
                    )
                except:
                    logger.warning(f"Failed to convert {col}, removing")
                    preprocessor._continuous_cols.remove(col)

    # Outlier handling
    if step_config.get('winsorize', True):
        logger.debug("  - Winsorizing continuous features and labels")
        preprocessor.winsorize_continuous()
        preprocessor.winsorize_label()

    # Target transformation
    if step_config.get('log_transform_target', True):
        logger.debug("  - Log-transforming target")
        preprocessor.log_label()

    # Drop problematic columns after split
    preprocessor._drop_problematic_cols_from_splits()

    # Imputation
    impute_method = step_config.get('impute_method', 'median')
    if impute_method == 'mice':
        logger.debug("  - Imputing with MICE")
        preprocessor.impute_missings_with_mice()
    elif impute_method in ['median', 'mean']:
        logger.debug(f"  - Imputing with {impute_method}")
        for col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].isnull().any():
                if pd.api.types.is_numeric_dtype(preprocessor.X_train[col]):
                    if impute_method == 'median':
                        fill_value = preprocessor.X_train[col].median()
                    else:  # mean
                        fill_value = preprocessor.X_train[col].mean()
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
                else:
                    fill_value = (
                        preprocessor.X_train[col].mode()[0]
                        if len(preprocessor.X_train[col].mode()) > 0
                        else 0
                    )
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
    elif impute_method == 'none':
        logger.debug("  - Skipping imputation")
    else:
        logger.warning(f"Unknown impute_method: {impute_method}, using median")
        # Use median as fallback
        for col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].isnull().any():
                if pd.api.types.is_numeric_dtype(preprocessor.X_train[col]):
                    fill_value = preprocessor.X_train[col].median()
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)

    # Normalization
    if step_config.get('normalize_continuous', True):
        logger.debug("  - Normalizing continuous features")
        preprocessor.normalize_continuous_cols()

    # Get processed data
    X_train = preprocessor.X_train
    y_train = preprocessor.y_train
    X_test = preprocessor.X_test
    y_test = preprocessor.y_test

    # Reset index for consistency with sample_train_set
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Fix string columns in final output
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                logger.warning(f"Converted column {col} from object to numeric")
            except:
                logger.warning(f"Could not convert column {col} to numeric, dropping")
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    logger.info(f"Preprocessing complete!")
    logger.info(f"Train pool shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Target log-transformed: {step_config.get('log_transform_target', True)}")

    return X_train, y_train, X_test, y_test, cbg_column
