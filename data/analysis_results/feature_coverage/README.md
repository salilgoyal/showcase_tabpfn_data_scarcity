# Data Filtering for Cross-County and Finetuning Experiments

This document explains how to use the data filtering functionality to restrict experiments to specific counties and features.

## Overview

The data filtering system allows you to:
1. **Filter counties**: Restrict experiments to a subset of counties (e.g., those with high feature coverage)
2. **Filter features**: Restrict to specific features (e.g., high-coverage features that appear in most counties)

Filtering is **optional** and can be easily enabled/disabled via configuration without code changes.

## Use Cases

### When to Use Filtering

1. **Feature Sparsity Issues**: When many features have low coverage across counties, causing NaN-heavy training data
2. **Data Quality**: Restrict to counties with complete data for more reliable model training
3. **Focused Analysis**: Test hypotheses about specific county subsets or feature sets
4. **Computational Efficiency**: Reduce data size by focusing on high-quality subsets

### Example: High-Coverage Filtering

Based on the feature investigation analysis, only 36 features have ≥50% coverage across counties, and only 87 counties have ALL these features. Using filtering to restrict to this high-quality subset:
- **87 counties** with complete feature coverage
- **36 features** present in all selected counties
- **~594K total data points** available
- **Zero NaN values** after preprocessing and alignment

## Configuration

### Config Structure

Add a `data_filtering` section to your experiment config file:

```yaml
data_filtering:
  counties:
    enabled: true          # Set to false to disable county filtering
    source: "file"         # "file" or "inline"
    file: "path/to/county_list.csv"  # Path to CSV with 'fips' column
    # OR use inline list:
    # list: [1001, 1003, 1005, ...]

  features:
    enabled: true          # Set to false to disable feature filtering
    source: "file"         # "file" or "inline"
    file: "path/to/feature_list.csv"  # Path to CSV with 'feature_name' column
    # OR use inline list:
    # list: ["char_bldg_sf", "char_yrblt", "sale_month", ...]
```

### File Format Requirements

**County list CSV** must have:
- A `fips` column containing integer FIPS codes
- Example: `output_preprocessed/high_coverage_county_list.csv`

**Feature list CSV** must have:
- A `feature_name` column containing feature names as strings
- Example: `output_preprocessed/high_coverage_features.csv`

## How to Enable/Disable Filtering

### To Enable Filtering

1. Uncomment the `data_filtering` section in your config file
2. Set `enabled: true` for counties and/or features
3. Specify file paths or inline lists

### To Disable Filtering

Either:
- Set `enabled: false` in the config
- Comment out the entire `data_filtering` section
- Remove the `data_filtering` section (defaults to no filtering)

### Partial Filtering

You can enable county filtering without feature filtering (or vice versa):

```yaml
data_filtering:
  counties:
    enabled: true
    source: "file"
    file: "path/to/county_list.csv"
  features:
    enabled: false  # No feature filtering
```

## Example Configurations

### Example 1: High-Coverage Counties and Features (Recommended)

```yaml
data_filtering:
  counties:
    enabled: true
    source: "file"
    file: "/home/users/salilg/tabpfn_data_scarcity/data/county_metadata/high_coverage_county_list.csv"
  features:
    enabled: true
    source: "file"
    file: "/home/users/salilg/tabpfn_data_scarcity/data/feature_lists/high_coverage_features.csv"
```

**Result**: 87 counties, 36 features, ~594K data points, no NaN values

### Example 2: County Filtering Only

```yaml
data_filtering:
  counties:
    enabled: true
    source: "file"
    file: "/home/users/salilg/tabpfn_data_scarcity/data/county_metadata/high_coverage_county_list.csv"
  features:
    enabled: false
```

**Result**: 87 counties, all available features (may still have some NaN due to feature alignment)

### Example 3: Inline County List

```yaml
data_filtering:
  counties:
    enabled: true
    source: "inline"
    list: [19153, 12117, 13117, 45083, 37097]  # Specific counties of interest
  features:
    enabled: false
```

### Example 4: No Filtering (Default Behavior)

```yaml
# No data_filtering section - all counties and features used
```

Or:

```yaml
data_filtering:
  counties:
    enabled: false
  features:
    enabled: false
```

## How Filtering Works Internally

### Cross-County Experiment

1. **County Filtering** (before sampling):
   - Filters metadata files before `SmallCountyInContextSampler` runs
   - Sampler only sees allowed counties
   - Counties are sampled from the filtered pool

2. **Feature Filtering** (after alignment):
   - Happens after feature alignment between train/test
   - Applied to both train and test sets
   - Preserves feature order from the allowed list

### Finetuning Experiment

1. **County Filtering** (at metadata load):
   - Filters `county_metadata` DataFrame immediately after loading
   - Only filtered counties considered for train/val/test splits
   - Applies to both regular and holdout counties

2. **Feature Filtering** (after alignment):
   - Happens after feature alignment between train/val/test
   - Applied to all three splits
   - Preserves feature order from the allowed list

## Files Provided

### Analysis Files (Input)

- `output_preprocessed/feature_coverage_preprocessed.csv` - Feature coverage statistics
- `output_preprocessed/county_feature_matrix_preprocessed.csv` - Binary matrix of feature presence

### Filter Lists (Output - Ready to Use)

- `output_preprocessed/high_coverage_county_list.csv` - 87 counties with all 36 high-coverage features
- `output_preprocessed/high_coverage_features.csv` - 36 features with ≥50% coverage

### Config Examples

- `experiments/configs/cross_county/small_in_context_10k.yaml` - Cross-county config with filtering template
- `experiments/configs/finetuning/large_scale.yaml` - Finetuning config with filtering template

## Validation and Error Handling

The filtering system includes validation:

### Warnings

- If no counties remain after filtering → logs warning
- If no features remain after filtering → raises error with details
- If allowed features not found in data → logs warning with list of missing features

### Errors

- Missing `fips` or `feature_name` column in filter files → raises ValueError
- File not found → raises FileNotFoundError
- Zero features after filtering → raises ValueError (prevents empty data)

## Performance Considerations

- **County filtering** is very fast (happens at metadata level)
- **Feature filtering** is fast (simple column selection after alignment)
- **No overhead** when filtering is disabled (default behavior preserved)

## Modifying Filter Lists

To create custom filter lists:

### Custom County List

```python
import pandas as pd

# Your custom FIPS codes
counties = [1001, 1003, 1005, 1007, ...]

df = pd.DataFrame({'fips': counties})
df.to_csv('my_custom_counties.csv', index=False)
```

### Custom Feature List

```python
import pandas as pd

# Your custom features
features = ['char_bldg_sf', 'char_yrblt', 'sale_month', ...]

df = pd.DataFrame({'feature_name': features})
df.to_csv('my_custom_features.csv', index=False)
```

## Troubleshooting

### Issue: No features remain after filtering

**Cause**: Allowed features don't match actual feature names after preprocessing

**Solution**:
1. Check `data/analysis_results/feature_coverage/feature_coverage_preprocessed.csv` for actual feature names
2. Ensure feature names match exactly (including case, underscores, suffixes)
3. Remember that preprocessing may add suffixes (e.g., `_miss`, `_cat_missing`)

### Issue: No counties remain after filtering

**Cause**: FIPS codes in filter list don't match available counties

**Solution**:
1. Check `data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv` for available FIPS codes
2. Ensure FIPS codes are integers, not zero-padded strings
3. Verify file path is correct and accessible

### Issue: Filtering not applied

**Cause**: Config section commented out or `enabled: false`

**Solution**:
1. Uncomment `data_filtering` section
2. Set `enabled: true` for relevant filters
3. Check logs for "County filtering enabled" / "Feature filtering enabled" messages

## Advanced: Programmatic Usage

You can use the `DataFilter` class directly in Python:

```python
from src.data import DataFilter

# Initialize from config
filter_config = {
    'counties': {
        'enabled': True,
        'source': 'file',
        'file': 'path/to/counties.csv'
    },
    'features': {
        'enabled': True,
        'source': 'inline',
        'list': ['feature1', 'feature2', 'feature3']
    }
}

data_filter = DataFilter(filter_config)

# Apply filtering
filtered_metadata = data_filter.filter_county_metadata(county_df)
filtered_features = data_filter.filter_features(X_train)

# Check status
if data_filter.is_county_filtering_enabled():
    print("County filtering active")
```

## Summary

The data filtering system provides a flexible, config-driven way to restrict experiments to high-quality data subsets. It's:

- ✅ **Optional**: Easily enabled/disabled without code changes
- ✅ **Modular**: Filter counties, features, or both independently
- ✅ **Flexible**: Support for file-based or inline filter lists
- ✅ **Safe**: Validation and error handling to prevent silent failures
- ✅ **Efficient**: No performance overhead when disabled

For the current use case (addressing feature sparsity), enabling both county and feature filtering with the high-coverage lists is recommended.
