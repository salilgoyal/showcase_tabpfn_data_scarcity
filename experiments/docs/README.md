# Preprocessing Configuration Guide

This guide explains how to configure the modular preprocessing pipeline for TabPFN experiments.

## Overview

The preprocessing system provides fine-grained control over:
1. **Feature selection** - Which feature categories to include
2. **Preprocessing steps** - Which transformations to apply

All configuration is done through `experiments/configs/base_config.yaml`.

## Quick Start

### Raw Data (No Preprocessing)

To test TabPFN with raw, unprocessed data:

```python
from src.data.loading import CountyDataLoader

loader = CountyDataLoader(
    county_csvs_dir='data/county_csvs',
    target_column='SALE_AMOUNT',
    preprocessing_config=None  # No preprocessing!
)
```

### Minimal Preprocessing

For basic cleaning only:

```yaml
preprocessing:
  features:
    property_chars: false
    census_bg: true
    assessed_value: true
    geographic: false
    temporal: false         # Don't use temporal features
  steps:
    generate_temporal_features: false  # No derived temporal vars
    drop_null_labels: true
    drop_single_value_cols: false
    drop_mostly_null_cols: false
    clean_bad_columns: false  # Keep all columns
    winsorize: false
    log_transform_target: false
    normalize_continuous: false
    impute_method: 'none'
```

### Full Preprocessing (Default)

The default config in `base_config.yaml` enables most preprocessing steps for comprehensive data preparation.

## Configuration Options

### Feature Selection (`preprocessing.features`)

Controls which feature categories to include:

```yaml
features:
  property_chars: true      # ~90-110 property characteristics (char_*)
  census_bg: true           # 15 census block group features
  census_tract: false       # 15 census tract features (alternative to BG)
  assessed_value: true      # CALCULATED_TOTAL_VALUE
  geographic: true          # latitude, longitude
  temporal: true            # Use temporal features (requires generate_temporal_features)
```

**Feature Counts:**
- Property chars: ~90-110 features (beds, baths, sqft, lot size, etc.)
- Census BG: 15 features (demographics, income, education, etc.)
- Census tract: 15 features (same as BG but at tract level)
- Assessed value: 1 feature
- Geographic: 2 features
- Temporal: 6 features (if generated from sale_date)

### Preprocessing Steps (`preprocessing.steps`)

#### Data Cleaning

```yaml
steps:
  drop_null_labels: true          # Drop rows with missing target
  drop_single_value_cols: true    # Drop columns with only one value
  drop_mostly_null_cols: true     # Drop columns with too many nulls
  share_non_null: 0.5             # Threshold for mostly_null (50%)
  drop_lowest_ratios: true        # Drop suspicious sales (lowest percentile)
  drop_repeat_sales: true         # Keep only most recent sale per property
  clean_bad_columns: true         # Remove problematic columns after train/test split
```

**`clean_bad_columns` explained:**
- After train/test split, some columns may become single-valued in one split
- Set to `false` if you want to test TabPFN with completely raw data
- Set to `true` for robust preprocessing

#### Feature Engineering

```yaml
steps:
  generate_temporal_features: true  # Generate time vars from sale_date
  one_hot_encode: true              # One-hot encode categoricals
```

**Temporal features generated (when enabled):**
- `sale_year` - Year of sale
- `sale_month` - Month (1-12)
- `sale_day_of_month` - Day of month (1-31)
- `sale_day_of_year` - Day of year (1-366)
- `sale_day_of_week` - Day of week (0-6)
- `sale_day` - Days since Jan 1, 2000

**To use only raw sale_date:** Set `generate_temporal_features: false`

#### Outlier Handling

```yaml
steps:
  winsorize: true             # Cap outliers at percentiles
  winsorize_percentile: 1     # Winsorize at 1st/99th percentile
```

Both continuous features AND target are winsorized when enabled.

#### Target Transformation

```yaml
steps:
  log_transform_target: true  # Apply log transformation to target
```

⚠️ **IMPORTANT:** When enabled, models train on `log(SALE_AMOUNT)`. The framework automatically inverse-transforms predictions, so all metrics are reported on the original price scale.

#### Feature Scaling

```yaml
steps:
  normalize_continuous: true  # StandardScaler normalization
```

Applies sklearn's StandardScaler to continuous features (fit on training set only).

#### Missing Value Imputation

```yaml
steps:
  impute_method: "median"  # Options: "median", "mean", "none"
```

- `"median"`: Fill with median (numeric) or mode (categorical)
- `"mean"`: Fill with mean (numeric) or mode (categorical)
- `"none"`: No imputation (leave NaNs - may cause errors)

## Common Configuration Patterns

### Pattern 1: Test TabPFN with Completely Raw Data

Goal: Evaluate how well TabPFN handles unprocessed data.

```yaml
preprocessing:
  features:
    property_chars: true
    census_bg: true
    assessed_value: true
    geographic: true
    temporal: false  # No temporal features at all
  steps:
    drop_null_labels: true  # Must drop these
    drop_single_value_cols: false
    drop_mostly_null_cols: false
    drop_lowest_ratios: false
    drop_repeat_sales: false
    clean_bad_columns: false
    generate_temporal_features: false
    one_hot_encode: false
    winsorize: false
    log_transform_target: false
    normalize_continuous: false
    impute_method: 'none'
```

### Pattern 2: Minimal Features with Full Preprocessing

Goal: Test preprocessing impact with small feature set.

```yaml
preprocessing:
  features:
    property_chars: false
    census_bg: true
    assessed_value: true
    geographic: false
    temporal: true
  steps:
    drop_null_labels: true
    drop_single_value_cols: true
    drop_mostly_null_cols: true
    share_non_null: 0.5
    drop_lowest_ratios: true
    drop_repeat_sales: true
    clean_bad_columns: true
    generate_temporal_features: true
    one_hot_encode: true
    winsorize: true
    winsorize_percentile: 1
    log_transform_target: true
    normalize_continuous: true
    impute_method: 'median'
```

### Pattern 3: Full Features with Conservative Preprocessing

Goal: Use all features but minimal transformations.

```yaml
preprocessing:
  features:
    property_chars: true
    census_bg: true
    assessed_value: true
    geographic: true
    temporal: true
  steps:
    drop_null_labels: true
    drop_single_value_cols: true
    drop_mostly_null_cols: true
    share_non_null: 0.5
    drop_lowest_ratios: false  # Keep all sales
    drop_repeat_sales: false   # Keep repeat sales
    clean_bad_columns: true
    generate_temporal_features: true
    one_hot_encode: true
    winsorize: false           # No outlier capping
    log_transform_target: false
    normalize_continuous: false
    impute_method: 'median'
```

### Pattern 4: Property Characteristics Only

Goal: Predict using only property-specific features.

```yaml
preprocessing:
  features:
    property_chars: true
    census_bg: false
    assessed_value: false
    geographic: false
    temporal: true  # Keep temporal for time trends
  steps:
    # Standard preprocessing
    drop_null_labels: true
    drop_single_value_cols: true
    drop_mostly_null_cols: true
    share_non_null: 0.5
    drop_lowest_ratios: true
    drop_repeat_sales: true
    clean_bad_columns: true
    generate_temporal_features: true
    one_hot_encode: true
    winsorize: true
    winsorize_percentile: 1
    log_transform_target: true
    normalize_continuous: true
    impute_method: 'median'
```

## Feature Set Comparison

| Configuration | Approx. Features | Use Case |
|--------------|------------------|----------|
| Property chars only | ~100 | Test property-specific prediction |
| Census + assessed | ~16 | Minimal viable feature set |
| All features | ~130 | Maximum information |
| Raw data (no temporal) | ~125 | Test with unprocessed data |

## Preprocessing Pipeline Order

The preprocessing steps are applied in this order:

```
1. Drop null labels (if enabled)
2. Generate temporal features (if enabled)
3. Drop single-value columns (if enabled)
4. Drop mostly-null columns (if enabled)
5. Drop lowest sales ratios (if enabled)
6. Drop repeat sales (if enabled)
7. One-hot encode categoricals (if enabled)
8. Train/test split
9. Clean bad columns from splits (if enabled)
10. Winsorize continuous features & target (if enabled)
11. Log-transform target (if enabled)
12. Impute missing values (if enabled)
13. Normalize continuous features (if enabled)
```

## Log Transformation Details

When `log_transform_target: true`:

1. **During preprocessing:** Target becomes `y = log(SALE_AMOUNT)`
2. **During training:** Model learns to predict log prices
3. **During prediction:** Model outputs log predictions
4. **During evaluation:** Framework automatically applies `exp()` to inverse-transform
5. **In results:** All metrics (MAE, RMSE, R²) are on original price scale

You don't need to manually handle the transformation.

## Troubleshooting

### Issue: "Column not found" errors

**Solution:** Ensure your county CSV files have the required columns:
- `SALE_AMOUNT` (target)
- `sale_date` (if using temporal features)
- Census variables (if using census features)
- Property characteristics (if using property_chars)

### Issue: Empty dataset after preprocessing

**Possible causes:**
1. Too many columns dropped by `drop_mostly_null_cols`
   - Solution: Lower `share_non_null` (e.g., 0.25)
2. All rows dropped by `drop_lowest_ratios`
   - Solution: Disable `drop_lowest_ratios`
3. County has very little data
   - Solution: Use a larger county or skip preprocessing

### Issue: Model performance differs significantly with/without preprocessing

**Expected behavior:** Log transformation and normalization can significantly affect model behavior. This is normal.

### Issue: Features missing in cross-county experiments

**Solution:** The framework automatically uses the intersection of features across counties. No action needed.

## Code Examples

### Using in Python Scripts

```python
from src.data.loading import CountyDataLoader

# Define preprocessing config
preprocessing_config = {
    'features': {
        'property_chars': True,
        'census_bg': True,
        'assessed_value': True,
        'geographic': True,
        'temporal': True,
    },
    'steps': {
        'generate_temporal_features': True,
        'drop_null_labels': True,
        'drop_single_value_cols': True,
        'drop_mostly_null_cols': True,
        'share_non_null': 0.5,
        'clean_bad_columns': True,
        'winsorize': True,
        'winsorize_percentile': 1,
        'log_transform_target': True,
        'normalize_continuous': True,
        'impute_method': 'median',
    }
}

# Initialize loader
loader = CountyDataLoader(
    county_csvs_dir='data/county_csvs',
    target_column='SALE_AMOUNT',
    preprocessing_config=preprocessing_config
)

# Load and preprocess data
df = loader.load_county(fips=17031)
X, y = loader.preprocess_for_training(df)

# Check if target is log-transformed
if loader.is_log_transformed():
    print("Target is log-transformed")
```

### Checking Configuration

```python
import yaml

# Load config
with open('experiments/configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check enabled features
features = config['preprocessing']['features']
enabled = [k for k, v in features.items() if v]
print(f"Enabled features: {enabled}")

# Check if log transform is enabled
log_transform = config['preprocessing']['steps']['log_transform_target']
print(f"Log transform: {log_transform}")
```

## Best Practices

1. **Start simple** - Begin with minimal features and basic preprocessing
2. **Compare systematically** - Change one setting at a time to understand impact
3. **Document your config** - Save the exact config used for each experiment
4. **Check feature counts** - Log the number of features to ensure consistency
5. **Validate results** - Check that metrics make sense for your domain

## Migration from Old Config Format

If you have old config files with:
```yaml
preprocessing:
  use_evelyn_preprocessing: true
  include_property_chars: false
```

Convert to new format:
```yaml
preprocessing:
  features:
    property_chars: false  # Was include_property_chars
    census_bg: true
    assessed_value: true
    geographic: true
    temporal: true
  steps:
    # ... all the processing steps
```

The old format is no longer supported.

## Additional Resources

### Related Files
- Config file: `experiments/configs/base_config.yaml`
- Loader implementation: `src/data/loading.py`
- Preprocessing implementation: `src/data/preprocessing.py`
- Column definitions: `src/data/column_categorizer.py`

### Example Configs
See `experiments/configs/` for example configurations for different experiment types.

## FAQ

**Q: How do I test TabPFN with completely raw data?**

A: Set `preprocessing_config=None` when creating CountyDataLoader, or set all cleaning steps to `false` in the config.

**Q: Can I generate only some temporal features?**

A: No - it's all or nothing with `generate_temporal_features`. Set to `false` to use only raw `sale_date`.

**Q: Why are there two winsorize functions?**

A: `winsorize_continuous()` handles features, `winsorize_label()` handles the target. Both use the same percentile and are controlled by one flag.

**Q: What happens if I disable imputation?**

A: Setting `impute_method: 'none'` leaves NaNs in the data, which may cause model training to fail.

**Q: Can I add custom preprocessing steps?**

A: Yes - modify `src/data/preprocessing.py` and add new flags to `step_config`.

**Q: How do I know if preprocessing worked correctly?**

A: Check the logs - preprocessing outputs detailed information about columns dropped, features generated, etc.

**Q: Does TabPFN work better with or without preprocessing?**

A: Test both! TabPFN is designed to handle raw data well, but preprocessing may still help in some cases.
