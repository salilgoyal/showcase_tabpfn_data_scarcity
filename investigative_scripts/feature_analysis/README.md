# Feature Coverage Analysis Scripts

This directory contains scripts for analyzing feature coverage across counties after preprocessing.

## Scripts

### `analyze_preprocessed_features.py`

Analyzes which features remain after preprocessing is applied to each county.

**Purpose**: Shows the real feature availability after the preprocessing pipeline (dropping single-value columns, mostly-null columns, etc.)

**Usage**:
```bash
python investigative_scripts/feature_analysis/analyze_preprocessed_features.py \
    <county_csvs_dir> \
    <output_dir> \
    <config_file>
```

**Example**:
```bash
python investigative_scripts/feature_analysis/analyze_preprocessed_features.py \
    /scratch/users/salilg/property_tax/county_csvs/ \
    data/analysis_results/feature_coverage/ \
    experiments/configs/cross_county/small_in_context_10k.yaml
```

**Outputs**:
- `feature_coverage_preprocessed.csv` - Coverage statistics for each feature
- `county_feature_matrix_preprocessed.csv` - Binary matrix of feature presence per county
- `coverage_summary_preprocessed.txt` - Human-readable summary

**SLURM**: Use `run_preprocessed_analysis.sh` to run on Sherlock (takes ~2-4 hours for all counties)

---

### `analyze_by_state.py`

Groups counties by state (first 2 digits of FIPS) and analyzes within-state feature consistency.

**Purpose**: Determines if feature heterogeneity is across-state vs within-state

**Usage**:
```bash
python investigative_scripts/feature_analysis/analyze_by_state.py \
    <county_feature_matrix_csv> \
    <output_dir>
```

**Example**:
```bash
python investigative_scripts/feature_analysis/analyze_by_state.py \
    data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv \
    data/analysis_results/feature_coverage/state_level/
```

**Outputs**:
- `state_feature_coverage.csv` - Feature coverage % for each state-feature pair
- `state_universal_features.csv` - Universal features per state
- `state_statistics.csv` - Summary stats per state
- `state_summary.txt` - Human-readable summary with insights

**Runtime**: Fast (~seconds), reads existing matrix

---

### `analyze_high_coverage_counties.py`

Identifies counties that have ALL high-coverage features and calculates total data availability.

**Purpose**: Find counties with complete feature sets for clean experiments without NaN

**Usage**:
```bash
python investigative_scripts/feature_analysis/analyze_high_coverage_counties.py \
    <feature_coverage_csv> \
    <county_matrix_csv> \
    <county_metadata_csv> \
    <output_file>
```

**Example**:
```bash
python investigative_scripts/feature_analysis/analyze_high_coverage_counties.py \
    data/analysis_results/feature_coverage/feature_coverage_preprocessed.csv \
    data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv \
    data/county_metadata/county_row_counts.csv \
    data/analysis_results/feature_coverage/high_coverage_analysis.txt
```

**Outputs**:
- `high_coverage_analysis.txt` - Detailed analysis with county list
- `high_coverage_county_list.csv` - FIPS codes and row counts

**Runtime**: Fast (~seconds), reads existing data

---

## Workflow

1. **Run preprocessing analysis** (expensive, run once):
   ```bash
   sbatch investigative_scripts/feature_analysis/run_preprocessed_analysis.sh
   ```
   This processes all ~2850 counties through the preprocessing pipeline.

2. **Analyze by state** (optional):
   ```bash
   python investigative_scripts/feature_analysis/analyze_by_state.py \
       data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv \
       data/analysis_results/feature_coverage/state_level/
   ```

3. **Find high-coverage counties**:
   ```bash
   python investigative_scripts/feature_analysis/analyze_high_coverage_counties.py \
       data/analysis_results/feature_coverage/feature_coverage_preprocessed.csv \
       data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv \
       data/county_metadata/county_row_counts.csv \
       data/analysis_results/feature_coverage/high_coverage_analysis.txt
   ```

## Results

All analysis results are saved to `data/analysis_results/feature_coverage/`:
- **Coverage summaries**: Feature-level and county-level statistics
- **Filter lists**: Counties and features for use in experiments
- **README**: Documentation on using data filtering

## Using Analysis Results

The analysis produces filter lists that can be used in experiments:

- **County filters**: `data/county_metadata/high_coverage_county_list.csv`
- **Feature filters**: `data/feature_lists/high_coverage_features.csv`

See `data/analysis_results/feature_coverage/README.md` for instructions on using these filters in experiments.

## Key Findings

From the most recent analysis:
- **143 raw features** exist in all counties
- After preprocessing: Only **36 features** have ≥50% coverage
- Only **87 counties** (3.1%) have all 36 high-coverage features
- These 87 counties contain **~594K data points** total
- Using these filters eliminates NaN values entirely

This explains the sparse matrices and NaN issues in the original experiments.
