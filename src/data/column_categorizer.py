"""
Column categorization for Cook County property data.

This module provides the single source of truth for column categorization,
enabling fine-grained feature selection via config flags.
"""

import pandas as pd
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# ============================================================================
# COLUMN PATTERNS
# ============================================================================

PROPERTY_CHAR_PATTERN = 'char_'
CENSUS_BG_PATTERN_SUFFIX = '_bg'
CENSUS_TRACT_PATTERN_SUFFIX = '_tract'


# ============================================================================
# EXPLICIT COLUMN LISTS
# ============================================================================

# Census block group columns
CENSUS_BG_COLUMNS = [
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

# Census tract columns
CENSUS_TRACT_COLUMNS = [
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
]

# Assessed value columns
ASSESSED_VALUE_COLUMNS = [
    'CALCULATED_TOTAL_VALUE',
    'ASSESSED_TOTAL_VALUE',
    'APPRAISED_TOTAL_VALUE',
    'MARKET_TOTAL_VALUE',
]

# Geographic columns
GEOGRAPHIC_COLUMNS = [
    'latitude',
    'longitude',
]

# Target column
TARGET_COLUMN = 'SALE_AMOUNT'

# Temporal source column (will be used to generate features)
TEMPORAL_SOURCE_COLUMN = 'sale_date'

# Temporal generated columns (created by gen_time_vars)
# These are typically auto-generated, not in raw data
TEMPORAL_GENERATED_COLUMNS = [
    'SALE_YEAR',
    'sale_month',
    'sale_day',
    'sale_quarter',
]

# Administrative/ID columns (kept as metadata, not used for modeling)
ADMINISTRATIVE_COLUMNS = [
    'Unnamed: 0',
    'ASSESSED_YEAR',
    'CENSUS_ID',
    'CLIP',
    'PREVIOUS_CLIP',
    'OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID',
    'address',
    'TOTAL_TAX_AMOUNT',
    'NET_TAX_AMOUNT',
    'TAX_RATE_AREA_CODE',
    'CALCULATED_TOTAL_VALUE_SOURCE_CODE',
    'fips',
    'tract',
    'block_group',
    'tract_id',
    'block_group_id',
    'MULTI_OR_SPLIT_PARCEL_CODE',
    'meta_sfh',
]


# ============================================================================
# COLUMN CATEGORIZER
# ============================================================================

class ColumnCategorizer:
    """Categorizes columns in a DataFrame into semantic groups."""

    def __init__(self, df: pd.DataFrame, target_column: str = TARGET_COLUMN):
        """
        Initialize categorizer.

        Args:
            df: DataFrame to categorize
            target_column: Name of target variable column
        """
        self.df = df
        self.target_column = target_column
        self.all_columns = set(df.columns)

    def categorize_all(self) -> Dict[str, List[str]]:
        """
        Categorize all columns in the DataFrame.

        Returns:
            Dictionary mapping category names to lists of column names:
            {
                'property_chars': [...],
                'census_bg': [...],
                'census_tract': [...],
                'assessed_value': [...],
                'geographic': [...],
                'temporal_source': ['sale_date'],
                'temporal_generated': [...],
                'target': ['SALE_AMOUNT'],
                'administrative': [...],
                'unknown': [...]  # columns that don't fit any category
            }
        """
        categories = {}
        categorized = set()

        # Target (highest priority)
        if self.target_column in self.all_columns:
            categories['target'] = [self.target_column]
            categorized.add(self.target_column)
        else:
            categories['target'] = []

        # Property characteristics (char_* pattern)
        property_chars = [col for col in self.all_columns
                         if col.startswith(PROPERTY_CHAR_PATTERN)]
        categories['property_chars'] = sorted(property_chars)
        categorized.update(property_chars)

        # Census block group
        census_bg = [col for col in CENSUS_BG_COLUMNS if col in self.all_columns]
        categories['census_bg'] = census_bg
        categorized.update(census_bg)

        # Census tract
        census_tract = [col for col in CENSUS_TRACT_COLUMNS if col in self.all_columns]
        categories['census_tract'] = census_tract
        categorized.update(census_tract)

        # Assessed value
        assessed_value = [col for col in ASSESSED_VALUE_COLUMNS if col in self.all_columns]
        categories['assessed_value'] = assessed_value
        categorized.update(assessed_value)

        # Geographic
        geographic = [col for col in GEOGRAPHIC_COLUMNS if col in self.all_columns]
        categories['geographic'] = geographic
        categorized.update(geographic)

        # Temporal source
        if TEMPORAL_SOURCE_COLUMN in self.all_columns:
            categories['temporal_source'] = [TEMPORAL_SOURCE_COLUMN]
            categorized.add(TEMPORAL_SOURCE_COLUMN)
        else:
            categories['temporal_source'] = []

        # Temporal generated (may or may not exist in raw data)
        temporal_generated = [col for col in TEMPORAL_GENERATED_COLUMNS
                             if col in self.all_columns]
        categories['temporal_generated'] = temporal_generated
        categorized.update(temporal_generated)

        # Administrative
        administrative = [col for col in ADMINISTRATIVE_COLUMNS if col in self.all_columns]
        categories['administrative'] = administrative
        categorized.update(administrative)

        # Unknown (columns that don't fit any category)
        unknown = sorted(self.all_columns - categorized)
        categories['unknown'] = unknown

        # Log summary
        logger.info(f"Column categorization complete:")
        logger.info(f"  Property chars: {len(categories['property_chars'])}")
        logger.info(f"  Census BG: {len(categories['census_bg'])}")
        logger.info(f"  Census tract: {len(categories['census_tract'])}")
        logger.info(f"  Assessed value: {len(categories['assessed_value'])}")
        logger.info(f"  Geographic: {len(categories['geographic'])}")
        logger.info(f"  Temporal source: {len(categories['temporal_source'])}")
        logger.info(f"  Temporal generated: {len(categories['temporal_generated'])}")
        logger.info(f"  Administrative: {len(categories['administrative'])}")
        logger.info(f"  Target: {len(categories['target'])}")
        if unknown:
            logger.warning(f"  Unknown: {len(unknown)} columns: {unknown[:10]}...")

        return categories


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_from_config(
    categories: Dict[str, List[str]],
    feature_config: Dict[str, bool]
) -> Dict[str, List[str]]:
    """
    Select features based on config flags.

    Args:
        categories: Output from ColumnCategorizer.categorize_all()
        feature_config: Dictionary with boolean flags:
            {
                'property_chars': bool,
                'census_bg': bool,
                'census_tract': bool,
                'assessed_value': bool,
                'geographic': bool,
                'temporal': bool,
            }

    Returns:
        Dictionary with selected column lists:
        {
            'continuous_cols': [...],
            'binary_cols': [...],
            'categorical_cols': [...],
            'meta_cols': [...]
        }
    """
    continuous_cols = []
    binary_cols = []
    categorical_cols = []

    # Property characteristics
    if feature_config.get('property_chars', False):
        property_chars = categories['property_chars']

        # Separate by suffix patterns
        for col in property_chars:
            if col.endswith('_miss'):
                # Missing indicators are binary
                binary_cols.append(col)
            elif col.endswith('_cat'):
                # Categorical features
                categorical_cols.append(col)
            else:
                # Default to continuous
                continuous_cols.append(col)

    # Census block group
    if feature_config.get('census_bg', False):
        continuous_cols.extend(categories['census_bg'])

    # Census tract
    if feature_config.get('census_tract', False):
        continuous_cols.extend(categories['census_tract'])

    # Assessed value
    if feature_config.get('assessed_value', False):
        # Only use CALCULATED_TOTAL_VALUE (primary assessed value column)
        if 'CALCULATED_TOTAL_VALUE' in categories['assessed_value']:
            continuous_cols.append('CALCULATED_TOTAL_VALUE')

    # Geographic
    if feature_config.get('geographic', False):
        continuous_cols.extend(categories['geographic'])

    # Temporal (source column + generated columns if they exist)
    if feature_config.get('temporal', True):  # Default True
        # Note: temporal_source will be used to generate features
        # The generated columns will be added during preprocessing
        # We include them here if they already exist in the data
        continuous_cols.extend(categories['temporal_generated'])

    # Meta columns (everything not used for modeling)
    meta_cols = (
        categories['administrative'] +
        categories['temporal_source'] +  # Keep sale_date as meta
        categories['target']  # Keep target as meta (handled separately)
    )

    # Remove duplicates while preserving order
    continuous_cols = list(dict.fromkeys(continuous_cols))
    binary_cols = list(dict.fromkeys(binary_cols))
    categorical_cols = list(dict.fromkeys(categorical_cols))
    meta_cols = list(dict.fromkeys(meta_cols))

    logger.info(f"Feature selection from config:")
    logger.info(f"  Continuous: {len(continuous_cols)}")
    logger.info(f"  Binary: {len(binary_cols)}")
    logger.info(f"  Categorical: {len(categorical_cols)}")
    logger.info(f"  Meta: {len(meta_cols)}")

    return {
        'continuous_cols': continuous_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'meta_cols': meta_cols,
    }


def get_feature_columns(
    df: pd.DataFrame,
    feature_config: Dict[str, bool],
    target_column: str = TARGET_COLUMN
) -> Dict[str, List[str]]:
    """
    One-stop convenience function: categorize columns and select features.

    Args:
        df: DataFrame to process
        feature_config: Feature selection flags
        target_column: Name of target column

    Returns:
        Dictionary with selected column lists:
        {
            'continuous_cols': [...],
            'binary_cols': [...],
            'categorical_cols': [...],
            'meta_cols': [...]
        }
    """
    categorizer = ColumnCategorizer(df, target_column)
    categories = categorizer.categorize_all()
    return select_features_from_config(categories, feature_config)
