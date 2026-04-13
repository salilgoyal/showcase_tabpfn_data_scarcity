# Quick Start Guide: 10K In-Context Experiment

## Overview

This experiment tests TabPFN's in-context learning ability by:
- Using 10K samples as in-context examples (no actual training)
- Testing on 10K held-out samples from small counties
- Running 10 iterations with different random splits

## Prerequisites

1. Data files exist:
   - `data/small_county_metadata.csv`
   - `data/county_row_counts.csv`
   - County CSV files in the configured directory

2. Environment is set up (conda or virtualenv with required packages)

## Quick Test (Local)

Before running the full experiment, test locally:

```bash
# 1. Activate environment
conda activate tabpfn_env  # or your environment name

# 2. Run integration tests
python notebooks/cross_county/test_sampler_integration.py

# 3. Run a quick test with small parameters
python experiments/run_experiment.py \
    --experiment_type cross_county \
    --config experiments/configs/cross_county/small_in_context_10k_test.yaml
```

The test config uses:
- Only 5 small counties
- 500 samples for train/test (instead of 10K)
- 1 iteration (instead of 10)
- Only XGBoost (for speed)

This should complete in a few minutes and verify the pipeline works.

## Run Full Experiment on Sherlock

### 1. Verify Paths in Config

Edit `experiments/configs/cross_county/small_in_context_10k.yaml` if needed:

```yaml
data:
  county_csvs_dir: "/scratch/users/salilg/property_tax/county_csvs/"
  county_metadata_file: "/home/users/salilg/tabpfn_data_scarcity/data/county_row_counts.csv"

output:
  results_dir: "/scratch/users/salilg/property_tax/results/cross_county/small_in_context_10k/"
```

### 2. Submit Job

```bash
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/small_in_context_10k.yaml
```

### 3. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with actual job ID)
tail -f logs/cross_county_JOBID.out
tail -f logs/cross_county_JOBID.err
```

## Expected Runtime

- **Test config**: ~5 minutes (local)
- **Full experiment**: ~2-6 hours (depends on hardware and preprocessing)
  - 10 iterations
  - 2 models (TabPFN + XGBoost)
  - XGBoost with 50 Optuna trials per iteration

## Output Files

Results will be saved to the configured output directory:

```
results/cross_county/small_in_context_10k/
├── results.csv              # Main results file
├── experiment.log           # Detailed logs
├── calibration.pkl          # Calibration data (if enabled)
└── predictions.parquet      # Predictions (if enabled)
```

## Key Metrics in Results

The `results.csv` file will contain:
- `model`: Model name (tabpfn, xgboost)
- `iteration`: Iteration number (0-9)
- `target_fips`: Will be "sampled" for this experiment
- `train_size`: Should be ~10K
- `test_size`: Should be ~10K
- `r2`, `mae`, `rmse`: Performance metrics
- `fit_time`, `pred_time`: Timing information

## Troubleshooting

### "County data not found"
- Check that `county_csvs_dir` path is correct
- Verify county FIPS codes exist in the metadata files

### "Sampling result not available"
- Ensure the `sampling` section is in your config
- Check that metadata files exist and are readable

### Memory issues
- Reduce `target_train_size` and `target_test_size` in config
- Reduce `n_small_counties` to use fewer counties

### Job timeout
- Increase `#SBATCH --time` in the SLURM script
- Reduce `optuna_trials` in the XGBoost config

## Modifying the Experiment

### Change sample sizes:
```yaml
sampling:
  parameters:
    target_train_size: 5000   # Instead of 10K
    target_test_size: 5000
```

### Change number of small counties:
```yaml
sampling:
  parameters:
    n_small_counties: 25      # Instead of 50
```

### Change iterations:
```yaml
experiment:
  repetitions: 5              # Instead of 10
iterations: 5
```

### Disable a model:
```yaml
models:
  - name: "tabpfn"
    enabled: false            # Disable TabPFN
  - name: "xgboost"
    enabled: true
```

## Next Steps After Experiment Completes

1. **Analyze results**:
   ```python
   import pandas as pd

   results = pd.read_csv('results/cross_county/small_in_context_10k/results.csv')

   # Average performance across iterations
   print(results.groupby('model')[['r2', 'mae', 'rmse']].mean())
   ```

2. **Compare to baseline**: Compare against within-county experiments

3. **Visualize**: Create plots of performance vs. iteration

4. **Debug**: Check `experiment.log` for any warnings or issues

## Questions?

See `notebooks/cross_county/INTEGRATION_SUMMARY.md` for detailed architecture information.
