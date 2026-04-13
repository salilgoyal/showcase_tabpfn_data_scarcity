# Feature Investigation Directory Reorganization

**Date**: January 15, 2026

## What Changed

The `notebooks/feature_investigation/` directory has been reorganized for better clarity and maintainability.

### Old Structure (Archived)
```
notebooks/feature_investigation/
├── analyze_*.py                  # Analysis scripts
├── run_*.sh                      # SLURM scripts
├── DATA_FILTERING_README.md      # Documentation
├── output_preprocessed/          # Analysis results
├── high_coverage_counties/       # Filter lists
├── output_by_state/              # State-level analysis
└── feature_name_overlap/         # Raw feature analysis
```

### New Structure
```
data/
├── county_metadata/
│   ├── county_row_counts.csv              # Existing
│   ├── small_county_metadata.csv          # Existing
│   └── high_coverage_county_list.csv      # NEW: Filter list
│
├── feature_lists/
│   └── high_coverage_features.csv         # NEW: Filter list
│
└── analysis_results/feature_coverage/
    ├── README.md                          # Documentation (moved)
    ├── feature_coverage_preprocessed.csv  # Analysis results
    ├── county_feature_matrix_preprocessed.csv
    ├── coverage_summary_preprocessed.txt
    ├── high_coverage_analysis.txt
    └── state_level/                       # State analysis results
        ├── state_feature_coverage.csv
        ├── state_universal_features.csv
        ├── state_statistics.csv
        └── state_summary.txt

scripts/analysis/
├── README.md                              # NEW: Script documentation
├── analyze_preprocessed_features.py       # Moved
├── analyze_by_state.py                    # Moved
├── analyze_high_coverage_counties.py      # Moved
└── run_preprocessed_analysis.sh           # Moved
```

## Rationale

**Before**: Mixed purposes - scripts, outputs, filter lists, and docs all in one place

**After**: Clear separation:
- `data/` = reusable data artifacts (inputs to experiments)
- `data/analysis_results/` = analysis outputs (for reference/documentation)
- `scripts/analysis/` = one-off analysis tools

## Updated File Paths

### Config Files

Experiment configs now reference the new paths:

```yaml
# Old
file: "/home/.../notebooks/feature_investigation/output_preprocessed/high_coverage_county_list.csv"

# New
file: "/home/.../data/county_metadata/high_coverage_county_list.csv"
```

Updated files:
- `experiments/configs/cross_county/small_in_context_10k.yaml`
- `experiments/configs/finetuning/large_scale.yaml`

### Documentation

- Main README: `data/analysis_results/feature_coverage/README.md`
- Script docs: `scripts/analysis/README.md`

## What to Do

If you need to reference the old structure, it's archived at:
```
archive/notebooks_feature_investigation_20260115/
```

For new work:
- **Filter lists**: Use files in `data/county_metadata/` and `data/feature_lists/`
- **Analysis results**: See `data/analysis_results/feature_coverage/`
- **Running analyses**: Use scripts in `scripts/analysis/`

## Benefits

1. **Clearer organization**: Data artifacts vs analysis tools vs results
2. **Better discoverability**: Obvious where to find filter lists
3. **Conventional structure**: Follows ML project best practices
4. **Easier maintenance**: Clear ownership and purpose for each directory
