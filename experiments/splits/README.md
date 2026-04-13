# Pre-Generated Splits

This directory contains pre-generated test and train splits for reproducible experiments.

## Why Pre-Generate Splits?

1. **Reproducibility**: Same exact splits across all experiment runs
2. **Comparability**: Guaranteed that train_v1 vs train_v2 vs train_v3 use identical test set
3. **Auditability**: Can inspect and validate splits before expensive training
4. **Efficiency**: Generate once, use many times

## Directory Structure

```
experiments/splits/
├── test_v1/                       # Test set v1 (generated once)
│   ├── test_indices.npy           # Row indices for test
│   ├── train_pool_indices.npy     # Row indices available for training
│   ├── test_counties.json         # List of test county FIPS
│   ├── county_info.json           # Per-county statistics
│   ├── size_buckets.json          # Counties in each size bucket
│   ├── metadata.json              # Split metadata
│   ├── summary_report.txt         # Human-readable summary
│   │
│   ├── train_v1/                  # Train set: test history only
│   │   ├── train_indices.npy
│   │   ├── source_breakdown.json
│   │   ├── county_distribution.json
│   │   ├── metadata.json
│   │   └── summary_report.txt
│   │
│   ├── train_v2/                  # Train set: mixed (50% history, 50% external)
│   │   ├── train_indices.npy
│   │   ├── source_breakdown.json
│   │   ├── county_distribution.json
│   │   ├── metadata.json
│   │   └── summary_report.txt
│   │
│   ├── train_v3/                  # Train set: external only
│   │   └── ...
│   │
│   ├── train_v4/                  # Train set: stratified external
│   │   └── ...
│   │
│   └── train_v5/                  # Train set: large-scale (all data)
│       └── ...
```

## Workflow

### Step 1: Generate Test Set (Once)

```bash
# Generate test set from preprocessed data
python experiments/scripts/generate_test_set.py \
    --config experiments/configs/test_sets/test_v1.yaml \
    --data_path /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet \
    --output_dir experiments/splits/test_v1/

# Inspect the test set
cat experiments/splits/test_v1/summary_report.txt
cat experiments/splits/test_v1/metadata.json
```

**What this does**:
- Selects 36 counties across 5 size buckets (tiny → xlarge)
- Within each county, splits by date (top 50% = test, bottom 50% = train pool)
- Saves indices to disk for reproducibility

### Step 2: Generate Train Sets (Multiple Strategies)

```bash
# Generate train_v1: test county history only
python experiments/scripts/generate_train_set.py \
    --config experiments/configs/train_sets/train_v1.yaml \
    --test_split_dir experiments/splits/test_v1/ \
    --data_path /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet \
    --output_dir experiments/splits/test_v1/train_v1/

# Generate train_v2: mixed (50% history, 50% external)
python experiments/scripts/generate_train_set.py \
    --config experiments/configs/train_sets/train_v2.yaml \
    --test_split_dir experiments/splits/test_v1/ \
    --data_path /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet \
    --output_dir experiments/splits/test_v1/train_v2/

# Generate train_v3: external only
python experiments/scripts/generate_train_set.py \
    --config experiments/configs/train_sets/train_v3.yaml \
    --test_split_dir experiments/splits/test_v1/ \
    --data_path /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet \
    --output_dir experiments/splits/test_v1/train_v3/

# Inspect train sets
cat experiments/splits/test_v1/train_v2/summary_report.txt
```

### Step 3: Run Experiments (Many Times)

```bash
# Run experiment with pre-generated splits
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml

# Run multiple training strategies to compare
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v1.yaml
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v3.yaml
```

## Experiment Config Format

To use pre-generated splits, your experiment config should specify split directories:

```yaml
# experiments/configs/cross_county/test_v1_train_v2.yaml
experiment:
  type: "cross_county"
  name: "test_v1_train_v2"

data:
  cleaned_data_path: "/scratch/.../cleaned_datasets/v1_no_onehot/"
  target_column: "SALE_AMOUNT"

# Use pre-generated splits (recommended)
splits:
  test_set_dir: "experiments/splits/test_v1/"
  train_set_dir: "experiments/splits/test_v1/train_v2/"

# Alternative: Generate on-the-fly (for development)
# test_set_config: "experiments/configs/test_sets/test_v1.yaml"
# train_set_config: "experiments/configs/train_sets/train_v2.yaml"

preprocessing:
  phase2_steps:
    winsorize: true
    # ...
```

## File Formats

### test_indices.npy / train_indices.npy
- NumPy arrays of integer row indices into the full dataset
- Can be loaded with `np.load()`

### JSON files
- Human-readable metadata
- County lists, statistics, breakdowns

### summary_report.txt
- Human-readable summary of the split
- Review before running experiments

## Tips

1. **Validate before training**: Always check `summary_report.txt` to ensure splits look correct

2. **Version control**: Commit metadata JSON files to git (not .npy files - too large)

3. **Checksums**: Record checksums of .npy files for verification:
   ```bash
   md5sum experiments/splits/test_v1/*.npy > experiments/splits/test_v1/checksums.txt
   ```

4. **Multiple test sets**: Can create multiple test sets (test_v1, test_v2) for different research questions

5. **Regenerate if needed**: If you discover issues, regenerate and document the change

## Troubleshooting

### Splits don't exist yet
If you try to run an experiment and splits don't exist:
```
FileNotFoundError: experiments/splits/test_v1/ not found
```

Solution: Generate the splits first (Steps 1-2 above)

### Wrong split version
If config points to non-existent split directory, update config or generate that version.

### Need to regenerate
To regenerate splits with different settings:
1. Modify config file
2. Delete old split directory
3. Re-run generation scripts
