# Data Organization Guide

This document describes how data, analysis results, and filter lists are organized in this project.

## Directory Structure

```
tabpfn_data_scarcity/
├── data/                          # Data artifacts and analysis results
│   ├── county_metadata/           # County metadata files
│   │   ├── county_row_counts.csv
│   │   ├── small_county_metadata.csv
│   │   └── high_coverage_county_list.csv      # Filter: 87 counties with complete features
│   │
│   ├── feature_lists/             # Feature filter lists
│   │   └── high_coverage_features.csv          # Filter: 36 high-coverage features
│   │
│   └── analysis_results/          # Analysis outputs (for reference)
│       └── feature_coverage/
│           ├── README.md                       # How to use data filtering
│           ├── feature_coverage_preprocessed.csv
│           ├── county_feature_matrix_preprocessed.csv
│           ├── coverage_summary_preprocessed.txt
│           ├── high_coverage_analysis.txt
│           └── state_level/                    # State-level analysis
│
├── investigative_scripts/         # One-off investigative analysis scripts
│   └── feature_analysis/          # Feature coverage analysis scripts
│       ├── README.md                           # Script documentation
│       ├── analyze_preprocessed_features.py
│       ├── analyze_by_state.py
│       ├── analyze_high_coverage_counties.py
│       └── run_preprocessed_analysis.sh
│
├── experiments/                   # Experiment configs and runners
│   ├── configs/
│   │   ├── cross_county/
│   │   └── finetuning/
│   ├── slurm/
│   └── scripts/
│       └── analysis/              # Experiment result aggregation scripts
│           └── aggregate_results.py
│
├── src/                          # Core library code
│   ├── data/                     # Data loading and filtering
│   │   ├── filters.py            # DataFilter class
│   │   └── ...
│   ├── models/
│   └── evaluation/
│
└── archive/                      # Archived/deprecated content
    └── notebooks_feature_investigation_20260115/
```

## Key Files and Their Purpose

### Data Artifacts (Inputs to Experiments)

| File | Location | Purpose |
|------|----------|---------|
| `county_row_counts.csv` | `data/county_metadata/` | Row counts for all counties |
| `small_county_metadata.csv` | `data/county_metadata/` | Metadata for small counties |
| `high_coverage_county_list.csv` | `data/county_metadata/` | **Filter**: 87 counties with complete features |
| `high_coverage_features.csv` | `data/feature_lists/` | **Filter**: 36 features with ≥50% coverage |

### Analysis Results (Outputs for Reference)

| File | Location | Purpose |
|------|----------|---------|
| `README.md` | `data/analysis_results/feature_coverage/` | Data filtering documentation |
| `feature_coverage_preprocessed.csv` | `data/analysis_results/feature_coverage/` | Coverage stats per feature |
| `county_feature_matrix_preprocessed.csv` | `data/analysis_results/feature_coverage/` | Binary matrix of feature presence |
| `high_coverage_analysis.txt` | `data/analysis_results/feature_coverage/` | Summary of high-coverage analysis |

### Analysis Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `README.md` | `investigative_scripts/feature_analysis/` | Script usage documentation |
| `analyze_preprocessed_features.py` | `investigative_scripts/feature_analysis/` | Analyze feature coverage after preprocessing |
| `analyze_by_state.py` | `investigative_scripts/feature_analysis/` | Group counties by state, analyze coverage |
| `analyze_high_coverage_counties.py` | `investigative_scripts/feature_analysis/` | Find counties with complete features |

## Common Tasks

### Using Data Filtering in Experiments

To restrict experiments to high-quality data (87 counties, 36 features, no NaN):

1. Edit your experiment config:
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

2. Run your experiment normally

See `data/analysis_results/feature_coverage/README.md` for complete documentation.

### Re-running Feature Analysis

If preprocessing logic changes, re-analyze feature coverage:

```bash
# On Sherlock (takes 2-4 hours)
sbatch investigative_scripts/feature_analysis/run_preprocessed_analysis.sh

# Then analyze high-coverage counties
python investigative_scripts/feature_analysis/analyze_high_coverage_counties.py \
    data/analysis_results/feature_coverage/feature_coverage_preprocessed.csv \
    data/analysis_results/feature_coverage/county_feature_matrix_preprocessed.csv \
    data/county_metadata/county_row_counts.csv \
    data/analysis_results/feature_coverage/high_coverage_analysis.txt
```

See `investigative_scripts/feature_analysis/README.md` for detailed instructions.

### Creating Custom Filter Lists

To create your own county or feature filters:

```python
import pandas as pd

# Custom county list
counties = [1001, 1003, 1005, ...]
df = pd.DataFrame({'fips': counties})
df.to_csv('data/county_metadata/my_custom_counties.csv', index=False)

# Custom feature list
features = ['char_bldg_sf', 'char_yrblt', ...]
df = pd.DataFrame({'feature_name': features})
df.to_csv('data/feature_lists/my_custom_features.csv', index=False)
```

Then reference in your experiment config:
```yaml
data_filtering:
  counties:
    file: "/home/.../data/county_metadata/my_custom_counties.csv"
  features:
    file: "/home/.../data/feature_lists/my_custom_features.csv"
```

## Design Principles

### Why This Organization?

1. **Clear Separation**:
   - `data/` = inputs and reference outputs
   - `scripts/` = tools for generating outputs
   - `src/` = reusable library code

2. **Conventional Structure**:
   - Follows ML project best practices
   - Easy for new collaborators to understand

3. **Maintainability**:
   - Filter lists clearly marked as "inputs"
   - Analysis results clearly marked as "reference"
   - Scripts separate from experiments

### data/ vs src/

- **`data/`**: CSVs, analysis results, filter lists (data artifacts)
- **`src/`**: Python modules, reusable code (library code)

### investigative_scripts/ vs experiments/scripts/

- **`investigative_scripts/`**: One-off investigative analyses (feature coverage, data exploration)
- **`experiments/scripts/`**: Scripts for processing experiment results (result aggregation, plotting)

## Migration Notes

The `notebooks/feature_investigation/` directory was reorganized on 2026-01-15.

- **Old location**: `notebooks/feature_investigation/`
- **Archived at**: `archive/notebooks_feature_investigation_20260115/`
- **New locations**: See structure above

See `archive/notebooks_feature_investigation_20260115/REORGANIZATION_NOTE.md` for details.

## Related Documentation

- **Data filtering**: `data/analysis_results/feature_coverage/README.md`
- **Analysis scripts**: `investigative_scripts/feature_analysis/README.md`
- **Reorganization notes**: `archive/notebooks_feature_investigation_20260115/REORGANIZATION_NOTE.md`
