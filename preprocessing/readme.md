# Preprocessing Pipeline

This directory contains the Phase 1 preprocessing pipeline for cleaning and preparing pooled county data.

## Overview

**Phase 1 Preprocessing** is applied once to the full pooled dataset and creates a cleaned parquet file that all experiments can use. This approach:
- Avoids data leakage (statistical transformations are done in Phase 2 per-split)
- Improves efficiency (preprocessing only done once)
- Ensures consistency across all experiments

## Directory Structure

```
preprocessing/
├── configs/           # Phase 1 preprocessing configurations
├── scripts/           # Preprocessing and analysis scripts
├── slurm/            # SLURM job scripts
├── logs/             # Job logs (created automatically)
└── analysis/         # Analysis outputs (created automatically)
```

## Quick Start

### 1. Run Phase 1 Preprocessing

Clean and prepare the full pooled dataset:

```bash
# Submit preprocessing job
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v1_no_onehot.yaml

# Monitor progress
tail -f preprocessing/logs/clean_*.out
```

**Output**: `/scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/`
- `data.parquet` - Cleaned dataset (~2-3GB)
- `metadata.json` - Dataset statistics and column info
- `config.yaml` - Configuration used
- `preprocessing_log.txt` - Detailed log

**Resources**: 128GB RAM, 8 CPUs, 4 hours

### 2. Analyze Preprocessed Data

Before running experiments, analyze the cleaned data:

```bash
# Submit analysis job
sbatch preprocessing/slurm/analyze_data.sh \
    /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet \
    preprocessing/analysis/v1_no_onehot/

# Monitor progress
tail -f preprocessing/logs/analyze_*.out
```

**Output**: `preprocessing/analysis/v1_no_onehot/`
- `county_statistics.csv` - Per-county row counts and target stats
- `feature_overlap.csv` - Feature availability across counties
- `feature_types.csv` - Feature distributions
- `summary_report.txt` - Text summary
- `*.png` - Visualization plots

**Resources**: 64GB RAM, 4 CPUs, 2 hours

### 3. Run Experiments

Once preprocessing is complete, run experiments that use the cleaned data:

```bash
# Cross-county generalization experiment
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/cross_county_v2.yaml
```

See `experiments/` directory for experiment details.

## What Phase 1 Does

1. **Load Raw Data**: Load all county CSVs from `/scratch/users/salilg/property_tax/county_csvs/`
2. **Drop Null Labels**: Remove rows with missing target values
3. **Drop Single-Value Columns**: Remove columns with only one unique value
4. **Drop Mostly-Null Columns**: Remove columns with >50% nulls
5. **Drop Lowest Ratios**: Remove suspicious transactions (lowest 5% sale price ratios)
6. **Drop Repeat Sales**: Keep only most recent sale per property
7. **Generate Temporal Features**: Extract year, month, day, etc. from `sale_date`
8. **Label Encode Categoricals**: Encode categorical columns as integers
9. **Log Transform Target**: Apply log to `SALE_AMOUNT`
10. **Filter Features**: Apply feature whitelist if enabled
11. **Save to Parquet**: Compress and save (~10-15GB → ~2-3GB)

## What Phase 1 Does NOT Do

Phase 1 avoids statistical transformations that could cause data leakage:

- ❌ No normalization/standardization (done in Phase 2 per-split)
- ❌ No winsorization (done in Phase 2 per-split)
- ❌ No imputation with train-dependent statistics (done in Phase 2 per-split)

These are handled by Phase 2 preprocessing in each experiment, which fits on training data only.

## Tips

1. **Preprocess once, experiment many times**: Phase 1 is slow but only needed once
2. **Check metadata.json**: Inspect preprocessing output to understand what was done
3. **Analyze before experimenting**: Run analysis to ensure preprocessing worked as expected
4. **Monitor memory**: 21M rows requires 128GB+ RAM for preprocessing
5. **Use parquet**: 10x faster loading than CSV (~10 seconds vs minutes)

## Future Migrations (Optional)

- Migrate `within_county.py` to use `CleanedDataLoader`
- Migrate `data_scaling.py` to use `CleanedDataLoader`
- Archive/delete truly obsolete files
