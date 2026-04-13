"""
Data loading and preparation utilities for Cook County experiments.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path, cbg_column='block_group_id'):
    """
    Load Cook County data and prepare for experiment.

    Steps:
    1. Load data
    2. Generate time variables from sale_date
    3. Drop repeat sales (keep last sale per property per year-month)
    4. Split into pre-2022 (train pool) and 2022 (test set)
    5. Drop non-feature columns (address, CLIP, identifiers)
    6. Drop object columns to avoid TabPFN encoding issues

    Args:
        data_path: Path to cook_county.csv
        cbg_column: Name of census block group column (default: 'block_group_id')

    Returns:
        X_train_pool, y_train_pool, X_test, y_test, cbg_column_name
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded data shape: {df.shape}")

    # Generate time variables from sale_date
    logger.info("Generating time variables from sale_date")
    sale_date = df['sale_date'].astype(str).str.zfill(8)
    sale_date = pd.to_datetime(sale_date, format='%Y%m%d', errors='coerce')

    df['sale_year'] = sale_date.dt.year
    df['sale_month'] = sale_date.dt.month
    df['sale_day_of_month'] = sale_date.dt.day
    df['sale_day_of_year'] = sale_date.dt.dayofyear
    df['sale_day_of_week'] = sale_date.dt.dayofweek

    # Days since Jan 1, 2000
    reference_date = pd.to_datetime('20000101', format='%Y%m%d')
    df['sale_day'] = (sale_date - reference_date).dt.days

    # Drop repeat sales (keep last sale per property per year-month)
    logger.info("Dropping repeat sales (keeping last sale per property per year-month)")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['CLIP', 'sale_year', 'sale_month'], keep='last')
    logger.info(f"Dropped {before_dedup - len(df)} repeat sales ({len(df)} remaining)")

    # Drop rows with missing SALE_AMOUNT
    df = df[~df['SALE_AMOUNT'].isnull()]
    logger.info(f"After dropping null SALE_AMOUNT: {len(df)} rows")

    # Split into train pool and test set (2022)
    train_pool = df[df['sale_year'] <= 2021].copy()
    # train_pool = df[df['sale_year'] == 2021].copy() # make training set only 2021
    test_set = df[df['sale_year'] == 2022].copy() # for Cook County this was the last year

    logger.info(f"Train pool (year 2021): {len(train_pool)} samples")
    logger.info(f"Test set (year 2022): {len(test_set)} samples")
    logger.info(f"Train pool years: {train_pool['sale_year'].min()}-{train_pool['sale_year'].max()}")

    # Prepare features: drop non-predictive columns
    cols_to_drop = [
        # 'address', 'CLIP', 'SALE_AMOUNT',
        # 'sale_date',  # already converted to numeric features
        # 'Unnamed: 0',  # index column from CSV if present

        # # # Target leakage columns
        # 'MARKET_TOTAL_VALUE',
        # 'ASSESSED_TOTAL_VALUE',
        # 'APPRAISED_TOTAL_VALUE',
        # 'TOTAL_TAX_AMOUNT',
        # 'NET_TAX_AMOUNT',
        # # don't drop 'CALCULATED_TOTAL_VALUE',

        # # # ID and administrative columns
        # 'PREVIOUS_CLIP',
        # 'OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID',
        # 'SALE_YEAR',
        # 'ASSESSED_YEAR',
        # 'TAX_YEAR',
        # 'CENSUS_ID',
        # 'fips',
        # 'tract',
        # 'block_group',
        # 'tract_id',
        # 'TAX_RATE_AREA_CODE',
        # 'CALCULATED_TOTAL_VALUE_SOURCE_CODE',
        # 'MULTI_OR_SPLIT_PARCEL_CODE',

        # temporal columns (just to experiment)
        # 'sale_year', 'sale_month', 'sale_day_of_month', 'sale_day_of_year', 'sale_day_of_week', 'sale_day'

        # test: DROPPING ALL COLUMNS EXCEPT CALCULATED_TOTAL_VALUE
        'Unnamed: 0',
        'ASSESSED_YEAR',
        'CENSUS_ID',
        # 'fips',
        'MARKET_TOTAL_VALUE',
        'latitude',
        'longitude',
        'address',
        # 'TAX_YEAR',
        'TOTAL_TAX_AMOUNT',
        'NET_TAX_AMOUNT',
        'TAX_RATE_AREA_CODE',
        # 'meta_sfh',
        # 'char_yrblt',
        # 'char_yrblt_miss',
        # 'char_air_central',
        # 'char_air_miss',
        # 'char_air_cat',
        # 'char_beds',
        # 'char_beds_miss',
        # 'char_bldg_sf',
        # 'char_bldg_sf_miss',
        # 'char_ground_sf',
        # 'char_ground_sf_miss',
        # 'char_second_sf',
        # 'char_second_sf_miss',
        # 'char_base_fin_sf',
        # 'char_base_fin_sf_miss',
        # 'char_base_unfin_sf',
        # 'char_base_unfin_sf_miss',
        # 'char_stories',
        # 'char_stories_miss',
        # 'char_bsmt_miss',
        # 'char_no_bsmt',
        # 'char_bsmt_cel',
        # 'char_bsmt_slab',
        # 'char_bsmt_cat',
        # 'car_bsmt_fin_miss',
        # 'char_bsmt_fin_par',
        # 'char_basement_unfin',
        # 'char_bsmt_fin_cat',
        # 'char_ext_wall_miss',
        # 'char_ext_wall_cat',
        # 'char_nbath',
        # 'char_nbath_miss',
        # 'char_fbath',
        # 'char_fbath_miss',
        # 'char_hbath',
        # 'char_hbath_miss',
        # 'char_has_frpl',
        # 'char_n_frpl',
        # 'char_n_frpl_miss',
        # 'char_gar1_cnst_miss',
        # 'char_gar1_cat',
        # 'char_gar1_size',
        # 'char_gar1_size_miss',
        # 'char_gar1_sf',
        # 'char_gar1_sf_miss',
        # 'char_has_pool',
        # 'char_floor_cat',
        # 'char_floor_miss',
        # 'char_bq_avg',
        # 'char_bq_blavg',
        # 'char_bq_econ',
        # 'char_bq_exc',
        # 'char_bq_fa',
        # 'char_bq_go',
        # 'char_bq_lo',
        # 'char_bq_lux',
        # 'char_bq_po',
        # 'char_bq_abavg',
        # 'char_bq_ordinal',
        # 'char_bq_miss',
        # 'char_bq_cat',
        # 'char_land_sf',
        # 'char_land_sf_miss',
        # 'char_frontage_sf',
        # 'char_frontage_sf_miss',
        # 'char_depth_sf',
        # 'char_depth_sf_miss',
        # 'char_heat_miss',
        # 'char_heat_cat',
        # 'char_roof_miss',
        # 'char_roof_material_cat',
        # 'char_style_miss',
        # 'char_style_cat',
        # 'char_found_miss',
        # 'char_found_cat',
        # 'char_fuel_typ_cat',
        # 'char_fuel_typ_miss',
        # 'char_sewer_cat',
        # 'char_sewer_miss',
        # 'char_water_cat',
        # 'char_water_miss',
        # 'char_elec_cat',
        # 'char_elec_miss',
        'CLIP',
        'SALE_AMOUNT',
        # 'SALE_YEAR',
        # 'MULTI_OR_SPLIT_PARCEL_CODE',
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
        # 'block_group_id'
    ]

    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in train_pool.columns]

    X_train_pool = train_pool.drop(columns=cols_to_drop, errors='ignore')
    y_train_pool = train_pool['SALE_AMOUNT']

    X_test = test_set.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_set['SALE_AMOUNT']

    # Verify CBG column exists
    if cbg_column not in X_train_pool.columns:
        raise ValueError(f"CBG column '{cbg_column}' not found. Available columns: {X_train_pool.columns.tolist()}")

    logger.info(f"Using CBG column: {cbg_column}")
    logger.info(f"Feature columns before object drop: {X_train_pool.shape[1]}")

    # Drop object (categorical) columns to avoid TabPFN encoding issues
    # Keep CBG column for now (will be used for subsetting)
    object_cols = X_train_pool.select_dtypes(include=['object']).columns.tolist()
    # Don't drop CBG column even if it's object type
    object_cols = [col for col in object_cols if col != cbg_column]

    if object_cols:
        logger.info(f"Dropping {len(object_cols)} object columns: {object_cols}")
        X_train_pool = X_train_pool.drop(columns=object_cols)
        X_test = X_test.drop(columns=object_cols)

    logger.info(f"Feature columns after object drop: {X_train_pool.shape[1]}")
    logger.info(f"Features: {X_train_pool.columns.tolist()[:10]}...")

    return X_train_pool, y_train_pool, X_test, y_test, cbg_column


def sample_train_set(X_train_pool, y_train_pool, train_size, seed):
    """
    Sample training data with specified size and random seed.

    Args:
        X_train_pool: Full training pool features
        y_train_pool: Full training pool targets
        train_size: Number of samples to draw
        seed: Random seed

    Returns:
        X_train, y_train
    """
    np.random.seed(seed)

    if train_size >= len(X_train_pool):
        logger.warning(f"Train size {train_size} >= pool size {len(X_train_pool)}, using all data")
        train_indices = X_train_pool.index
    else:
        train_indices = np.random.choice(X_train_pool.index, size=train_size, replace=False)

    X_train = X_train_pool.loc[train_indices]
    y_train = y_train_pool.loc[train_indices]

    return X_train, y_train


def get_train_cbgs(X_train, cbg_column):
    """
    Get unique census block groups from training set.

    Args:
        X_train: Training features
        cbg_column: Name of CBG column

    Returns:
        Set of unique CBG values
    """
    train_cbgs = set(X_train[cbg_column].dropna().unique())
    return train_cbgs


def subset_test_by_cbg(X_test, y_test, train_cbgs, cbg_column):
    """
    Subset test set to only include CBGs that appear in training set.

    Args:
        X_test: Full test features
        y_test: Full test targets
        train_cbgs: Set of CBG values from training set
        cbg_column: Name of CBG column

    Returns:
        X_test_cbg, y_test_cbg (subsetted to matching CBGs)
    """
    # Filter to rows where CBG is in training CBGs
    mask = X_test[cbg_column].isin(train_cbgs)
    X_test_cbg = X_test[mask].copy()
    y_test_cbg = y_test[mask].copy()

    logger.info(f"CBG-matched test set: {len(X_test_cbg)} / {len(X_test)} samples "
                f"({len(X_test_cbg)/len(X_test)*100:.1f}%) from {len(train_cbgs)} CBGs")

    return X_test_cbg, y_test_cbg


def prepare_features_for_model(X, cbg_column, keep_cbg=False):
    """
    Prepare features for model training/prediction.

    Args:
        X: Feature dataframe
        cbg_column: Name of CBG column
        keep_cbg: If False, drop CBG column before modeling

    Returns:
        X with CBG column optionally removed
    """
    if not keep_cbg and cbg_column in X.columns:
        X = X.drop(columns=[cbg_column])
        logger.debug(f"Dropped CBG column for modeling")

    return X
