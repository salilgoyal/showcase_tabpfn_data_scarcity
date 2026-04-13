# Preprocessing Options Guide

This guide explains how to use the two preprocessing pipelines available in the experiments framework.

## Overview

The experiments framework now supports two preprocessing approaches:

1. **Original Preprocessing** (default) - Simple, minimal transformations
2. **Evelyn's Preprocessing** - Comprehensive pipeline with winsorization, log transformation, and normalization

## Quick Start

### Enable Evelyn's Preprocessing

Edit `experiments/config/base_config.yaml`:

```yaml
preprocessing:
  use_evelyn_preprocessing: true  # Enable Evelyn's pipeline
  include_property_chars: false   # Use minimal features
```

Or pass as command-line override (if implemented in launcher scripts).

### Feature Set Options

When using Evelyn's preprocessing, you can choose between two feature sets:

- **Minimal** (`include_property_chars: false`): ~21 features
  - `CALCULATED_TOTAL_VALUE` (assessed value)
  - 15 census block group variables
  - 6 temporal features (year, month, day of year, etc.)

- **Full** (`include_property_chars: true`): ~110 features
  - All minimal features above
  - Property characteristics: beds, baths, building square footage, lot size, etc.
  - Latitude/longitude

## Preprocessing Comparison

| Aspect | Original | Evelyn's |
|--------|----------|----------|
| **Target transformation** | None | **Log transform** |
| **Feature normalization** | None | StandardScaler |
| **Outlier handling** | None | Winsorization (1st/99th percentile) |
| **Missing values** | Median imputation | Median/mode imputation |
| **Categorical encoding** | Dropped | One-hot encoding |
| **Feature selection** | Drop admin columns | Curated minimal set |
| **Temporal features** | None | Generated from sale_date |

## Critical Difference: Log Transformation

⚠️ **IMPORTANT**: When using Evelyn's preprocessing, the target variable (`SALE_AMOUNT`) is **log-transformed**.

### What This Means

- The model trains on `log(SALE_AMOUNT)` instead of raw prices
- The framework automatically inverse-transforms predictions before computing metrics
- Metrics (R², MAE, RMSE) are reported on the **original price scale**
- You don't need to manually handle the transformation

### How It Works

The log transformation is handled automatically:

1. **During preprocessing**: `y = log(SALE_AMOUNT)`
2. **During prediction**: Model predicts `log(price)`
3. **During evaluation**: Framework applies `exp()` to both predictions and targets
4. **In results**: All metrics are on the original price scale

### Code Implementation

In the runner scripts:

```python
# Flag is set during initialization
self.log_transformed = config.get('preprocessing', {}).get('use_evelyn_preprocessing', False)

# Metrics computation automatically handles inverse transform
metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)
```

## Detailed Preprocessing Steps

### Original Preprocessing

```
1. Drop rows with missing target
2. Exclude administrative columns (CLIP, fips, etc.)
3. Drop object/string columns
4. Drop all-NaN columns
5. Fill remaining NaNs with column median
```

### Evelyn's Preprocessing

```
1. Drop rows with missing target
2. Generate temporal features from sale_date
3. Drop single-value columns
4. Drop mostly-null columns (< 50% non-null)
5. Drop lowest sales ratio percentile (outlier detection)
6. Drop repeat sales (keep most recent)
7. One-hot encode categorical variables
8. Train/test split (to fit transformations on train only)
9. Winsorize continuous features at 1st/99th percentile
10. Winsorize target at 1st/99th percentile
11. **Log-transform target: y = log(y)**
12. Median/mode imputation for missing values
13. Normalize continuous features with StandardScaler
14. Recombine train/test for cross-validation
```

## Feature Details

### Minimal Feature Set (Evelyn's Default)

**Assessed Value** (1 feature):
- `CALCULATED_TOTAL_VALUE` - County's assessed property value

**Census Block Group Variables** (15 features):
- `census_pct_children_bg` - Percent children
- `census_pct_senior_bg` - Percent seniors
- `census_med_age_bg` - Median age
- `census_pct_married_hh_bg` - Percent married households
- `census_pct_single_hh_bg` - Percent single households
- `census_pct_high_school_bg` - Percent high school educated
- `census_pct_college_bg` - Percent college educated
- `census_pct_graduate_bg` - Percent graduate degree
- `census_pct_poverty_bg` - Percent below poverty line
- `census_med_hh_inc_bg` - Median household income
- `census_med_per_cap_inc_bg` - Median per capita income
- `census_pct_snap_bg` - Percent receiving SNAP benefits
- `census_unemp_rate_bg` - Unemployment rate
- `census_med_yr_built_bg` - Median year built
- `census_pct_renter_occ_bg` - Percent renter occupied

**Temporal Features** (6 features, auto-generated):
- `sale_year` - Year of sale
- `sale_month` - Month of sale (1-12)
- `sale_day_of_month` - Day of month (1-31)
- `sale_day_of_year` - Day of year (1-366)
- `sale_day_of_week` - Day of week (0-6)
- `sale_day` - Days since Jan 1, 2000

### Full Feature Set (Optional)

All minimal features plus:

**Property Characteristics** (~90 features):
- `char_beds_*` - Number of bedrooms (one-hot encoded)
- `char_baths_*` - Number of bathrooms (one-hot encoded)
- `char_bldg_sf` - Building square footage
- `char_lot_sf` - Lot square footage
- `char_year_built` - Year built
- `char_units_count` - Number of units
- `latitude`, `longitude` - Geographic coordinates
- Many other property-specific features

## Usage Examples

### Example 1: Run with Original Preprocessing

```bash
# Config: preprocessing.use_evelyn_preprocessing = false (default)
python experiments/runners/within_county_runner.py \
  --fips 6001 \
  --bin_name small \
  --k_folds 5
```

### Example 2: Run with Evelyn's Preprocessing (Minimal Features)

```bash
# Config: preprocessing.use_evelyn_preprocessing = true
#         preprocessing.include_property_chars = false
python experiments/runners/within_county_runner.py \
  --fips 6001 \
  --bin_name small \
  --k_folds 5
```

### Example 3: Run with Evelyn's Preprocessing (Full Features)

```bash
# Config: preprocessing.use_evelyn_preprocessing = true
#         preprocessing.include_property_chars = true
python experiments/runners/within_county_runner.py \
  --fips 6001 \
  --bin_name small \
  --k_folds 5
```

## Configuration Files

### Base Config (`config/base_config.yaml`)

```yaml
preprocessing:
  use_evelyn_preprocessing: false
  include_property_chars: false
```

This controls the preprocessing globally for all experiments.

### Experiment-Specific Overrides

You can override preprocessing settings in `within_county_config.yaml` or `cross_county_config.yaml`:

```yaml
# within_county_config.yaml
preprocessing:
  use_evelyn_preprocessing: true
  include_property_chars: false
```

## Implementation Details

### File Structure

```
experiments/
├── config/
│   └── base_config.yaml           # Preprocessing settings
├── data/
│   ├── loaders.py                 # CountyDataLoader with preprocessing
│   └── evelyn_preprocessing.py    # Evelyn's preprocessing wrapper
├── evaluation/
│   └── metrics.py                 # Metrics with log_transformed flag
└── runners/
    ├── within_county_runner.py    # Updated to use preprocessing config
    └── cross_county_runner.py     # Updated to use preprocessing config
```

### Key Classes and Methods

**`CountyDataLoader`** (`experiments/data/loaders.py`):
```python
loader = CountyDataLoader(
    county_csvs_dir="/path/to/csvs",
    target_column="SALE_AMOUNT",
    use_evelyn_preprocessing=True,  # Toggle here
    include_property_chars=False
)

X, y = loader.preprocess_for_training(df)
# If use_evelyn_preprocessing=True, y is log-transformed
```

**`compute_metrics`** (`experiments/evaluation/metrics.py`):
```python
metrics = compute_metrics(
    y_true=y_test,
    y_pred=predictions,
    log_transformed=True  # Set to True if using Evelyn's preprocessing
)
# Automatically applies exp() to inverse-transform before computing metrics
```

## Comparing Results Across Preprocessing Methods

When comparing results from different preprocessing approaches:

1. **Metrics are comparable** - Both methods report on original price scale
2. **Feature counts differ** - Original has more features (keeps all numeric columns)
3. **Model behavior may differ** - Log transformation changes the learning objective
4. **Performance may vary** - Evelyn's preprocessing often improves performance through better outlier handling

### Example Comparison

| Preprocessing | Features | MAE | R² |
|--------------|----------|-----|-----|
| Original | 150 | $45,000 | 0.65 |
| Evelyn (Minimal) | 21 | $38,000 | 0.72 |
| Evelyn (Full) | 110 | $36,000 | 0.74 |

*These are example numbers - actual results will vary by county*

## Troubleshooting

### Issue: Different metrics between preprocessing methods

**Expected behavior** - Log transformation changes the optimization objective, which can lead to different model behavior and metrics.

### Issue: "Column not found" errors with Evelyn preprocessing

**Solution**: Ensure county CSV files have the required columns:
- `CALCULATED_TOTAL_VALUE`
- Census block group variables (`census_*_bg`)
- `sale_date`

### Issue: Feature count mismatch between train and test

**Solution**: This is handled automatically in cross-county experiments through feature alignment. The framework uses the intersection of features.

### Issue: Negative R² scores

**Possible causes**:
1. Model predicting poorly (check for errors)
2. Test set is very different from train set
3. Not enough training data

Check that `log_transformed` flag is set correctly if using Evelyn's preprocessing.

## References

### Related Files

- Cook County analysis: `cook_county_analysis/src/evelyn_preprocessing.py`
- Original preprocessing: `evelyn_files/preprocess.py`
- Config template: `evelyn_files/census_loop_config.yaml`

### Related Documentation

- Cook County quick start: `cook_county_analysis/docs/QUICK_START_EVELYN_PREPROCESSING.md`
- Integration guide: `cook_county_analysis/docs/evelyn_preprocessing_integration.md`

## Best Practices

1. **Start with minimal features** - Use `include_property_chars=false` to test Evelyn's preprocessing
2. **Compare both methods** - Run experiments with both preprocessing pipelines
3. **Document your choice** - Note which preprocessing was used in your results
4. **Check convergence** - Some models may need more iterations with normalized features
5. **Monitor feature counts** - Log the number of features to ensure consistency

## FAQ

**Q: Which preprocessing should I use?**

A: Start with Evelyn's preprocessing (minimal features) as it's been validated on the full dataset. Compare with original if results seem unexpected.

**Q: Can I use log transformation without the other preprocessing steps?**

A: Not directly through config, but you can modify `CountyDataLoader._preprocess_original()` to add log transformation.

**Q: Do I need to change my model hyperparameters?**

A: You may want to retune hyperparameters, especially for XGBoost, as normalized features can affect optimal learning rates and tree depths.

**Q: How do I know if my results are on log scale or original scale?**

A: If you're using `compute_metrics()` from the framework, results are **always on original scale** (the framework handles inverse transformation automatically).

**Q: Can I mix preprocessing methods in one experiment?**

A: No - preprocessing is set per experiment run. Run separate experiments to compare methods.

## Version History

- **v1.0** (2024-11): Initial integration of Evelyn's preprocessing into experiments framework
  - Added toggle in config files
  - Automatic log transformation handling
  - Support for minimal and full feature sets
