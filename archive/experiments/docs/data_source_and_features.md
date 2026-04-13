# Data Source and Feature Information

## Data Source

The experiments use county-level real estate data stored at:
- **Location**: `/nlp/scr/salilg/county_csvs/`
- **Format**: Individual CSV files per county (e.g., `fips_1001.csv`, `fips_1003.csv`)
- **Metadata**: `/sailhome/salilg/tabpfn_data_scarcity/county_row_counts.csv`
- **Target Variable**: `SALE_AMOUNT`

This is configured in `experiments/config/base_config.yaml`.

## Enhanced County Metadata

The registry script (`00_create_county_registry.py`) creates an **enhanced metadata file** that includes:

### From Original `county_row_counts.csv`:
- `fips` - County FIPS code
- `filename` - CSV filename
- `row_count` - Number of observations
- `file_size_bytes` - File size in bytes
- `file_size_mb` - File size in MB

### Added by Registry Script:
- `bin_name` - Size bin assignment (e.g., "small")
- `k_folds` - Number of CV folds for this bin
- `num_features` - Number of features/columns in county data
- `feature_list` - Comma-separated list of all feature names

The enhanced metadata is saved to `/sailhome/salilg/tabpfn_data_scarcity/small_county_metadata.csv`.

## Feature Consistency Analysis

The registry script automatically checks feature consistency across all small counties:

1. **Loads each small county** to examine its features
2. **Compares feature sets** across counties
3. **Reports**:
   - Number of common features across ALL counties
   - Total unique features across ALL counties
   - Whether all counties have identical features
   - Which counties (if any) have different feature sets

### Example Output:

```
Feature Consistency Analysis:
  Common features across all counties: 128
  Total unique features: 128
  ✓ All counties have identical features!
```

Or if there are inconsistencies:

```
Feature Consistency Analysis:
  Common features across all counties: 125
  Total unique features: 128
  ⚠ Feature sets differ across counties!
  3 features are not present in all counties
  15 counties have different feature sets
```

## Handling Feature Mismatches in Pooled Experiments

If counties have different features, the **cross-county runner** automatically:

1. **Identifies common features** between train pool and test set
2. **Uses only the intersection** of features
3. **Logs warnings** if feature mismatches are detected:
   ```
   WARNING - Feature mismatch detected! Missing in test: 2, Missing in train: 1
   ```
4. **Reports final feature count** used for training

This ensures the experiments can proceed even if feature sets differ slightly, but you'll be aware of any data quality issues.

## Why This Matters

**For Within-County Experiments**: Feature mismatches don't matter since we train/test on the same county.

**For Cross-County (Pooled) Experiments**: Feature consistency is critical because:
- We're pooling data from multiple counties
- Models need consistent feature sets across train and test
- Missing features would cause errors or bias

If features differ substantially across counties, you may want to:
1. Investigate why (data collection issues? different years?)
2. Consider only pooling counties with identical features
3. Implement feature alignment/imputation strategies

## Running Feature Check

To see feature information without running experiments:

```bash
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts
python 00_create_county_registry.py
```

This will display:
- Data source paths
- Number of small counties found
- Feature consistency analysis
- Enhanced metadata with all columns

Then inspect `small_county_metadata.csv` to see feature counts per county.
