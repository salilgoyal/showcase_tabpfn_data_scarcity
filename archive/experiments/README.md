# Small County Experiments

This directory contains code to run experiments comparing TabPFN and XGBoost performance on small counties (10-100 observations).

## Directory Structure

```
experiments/
├── config/                  # Configuration files
│   ├── base_config.yaml
│   ├── within_county_config.yaml
│   └── cross_county_config.yaml
├── data/                    # Data utilities
│   ├── county_registry.py
│   ├── loaders.py
│   └── splitters.py
├── models/                  # Model wrappers
│   ├── base_model.py
│   ├── tabpfn_wrapper.py
│   └── xgboost_wrapper.py
├── evaluation/              # Evaluation utilities
│   ├── metrics.py
│   └── aggregation.py
├── runners/                 # Experiment runners
│   ├── within_county_runner.py
│   └── cross_county_runner.py
├── scripts/                 # Execution scripts
│   ├── 00_create_county_registry.py
│   ├── 02_launch_within_county_nlprun.sh
│   ├── 03_launch_cross_county_nlprun.sh
│   ├── 04_aggregate_results.py
│   ├── slurm_01_within_county.sh
│   └── slurm_02_cross_county.sh
└── docs/
    └── nlprun_commands.md   # Documentation for running experiments
```

## Quick Start

### 1. Create County Registry

Identify all small counties (10-100 observations):

```bash
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts
python 00_create_county_registry.py
```

This creates `small_county_metadata.csv` with all eligible counties.

### 2. Run Experiments

**Option A: Using nlprun (Recommended)**

See `docs/nlprun_commands.md` for detailed instructions.

Quick start:
```bash
# Within-county experiment
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts
bash 02_launch_within_county_nlprun.sh --max-parallel 10

# Cross-county experiment
bash 03_launch_cross_county_nlprun.sh --max-parallel 10
```

**Option B: Using SLURM**

```bash
# Within-county experiment
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts
NUM_COUNTIES=$(tail -n +2 ../small_county_metadata.csv | wc -l)
sbatch --array=0-$((NUM_COUNTIES-1)) slurm_01_within_county.sh

# Cross-county experiment
NUM_JOBS=$((NUM_COUNTIES * 10))  # 10 iterations per county
sbatch --array=0-$((NUM_JOBS-1)) slurm_02_cross_county.sh
```

### 3. Aggregate Results

After all jobs complete:

```bash
cd /sailhome/salilg/tabpfn_data_scarcity/experiments/scripts

# Aggregate within-county results
python 04_aggregate_results.py --experiment within_county

# Aggregate cross-county results
python 04_aggregate_results.py --experiment cross_county
```

### 4. Analyze Results

Results are saved to `/sailhome/salilg/tabpfn_data_scarcity/results/`:

- `within_county/within_county_fold_results.csv` - All fold-level results
- `within_county/within_county_county_aggregated.csv` - Per-county aggregates
- `within_county/within_county_bin_aggregated.csv` - Per-bin aggregates
- `within_county/within_county_overall_aggregated.csv` - Overall comparison

Similar files for `cross_county/`.

Use the Jupyter notebooks in `/sailhome/salilg/tabpfn_data_scarcity/notebooks/` for visualization.

## Experiments

### Experiment 1: Within-County Performance

**Goal**: Evaluate how well each model performs when trained and tested on the same county.

**Method**: Repeated K-fold cross-validation with nested hyperparameter tuning
- For each county:
  - R=10 repetitions × K=5 folds = 50 train/test splits
  - For each split, tune hyperparameters on training data, then evaluate on test data
  - Aggregate across all splits for robust estimate

**Configuration**: `config/within_county_config.yaml`

### Experiment 2: Cross-County Generalization

**Goal**: Evaluate how well models generalize when trained on pooled data from multiple counties.

**Method**: Hold-out one county's samples as test, train on rest of pooled data
- For each county:
  - T=10 iterations with different test samples
  - Each iteration: sample 20% of county as test, pool rest with all other counties for training
  - Tune hyperparameters on each pooled training set
  - Evaluate generalization to held-out county samples

**Configuration**: `config/cross_county_config.yaml`

## Configuration

Edit `config/base_config.yaml` to adjust:
- County size bins (min/max observations)
- Number of repetitions/iterations
- Optuna hyperparameter search settings
- Resource requirements

## Computational Requirements

### Within-County Experiment
- **Per county**: ~2-4 hours on 1 GPU, 8 CPUs, 32GB RAM
- **Total**: ~50-200 jobs (one per small county)

### Cross-County Experiment
- **Per (county, iteration)**: ~1-3 hours on 1 GPU, 8 CPUs, 64GB RAM
- **Total**: ~500-2000 jobs (10 iterations × num_counties)

## Monitoring

Check running jobs:
```bash
nlpjobs  # or squeue -u $USER
```

Check logs:
```bash
tail -f /sailhome/salilg/tabpfn_data_scarcity/logs/within_county_*.out
```

## Troubleshooting

**Problem**: No small counties found

**Solution**: Check `county_row_counts.csv` has counties with 10-100 observations. Adjust bin ranges in `config/base_config.yaml` if needed.

**Problem**: Out of memory errors

**Solution**: Increase memory in SLURM/nlprun commands (try 64GB or 128GB)

**Problem**: Jobs failing on specific counties

**Solution**: Check logs for that county. May have data quality issues. Can manually exclude problem counties from metadata file.

## Adding New Experiments

The code is modular and extensible:

1. **New size bins**: Edit `config/base_config.yaml` to add more bins
2. **New models**: Create wrapper in `models/` inheriting from `BaseModel`
3. **New experiments**: Create runner in `runners/` using existing utilities
4. **New metrics**: Add to `evaluation/metrics.py`

## Citation

If you use this code, please cite:
[Add citation information]
