# TabPFN Data Scarcity Experiments

Compare TabPFN and XGBoost performance on property tax assessment data across different data regimes, with focus on data scarcity scenarios.

## Overview

This repository implements experiments comparing TabPFN (Tabular Prior-Fitted Network) and XGBoost on property sale price prediction using county-level transaction data. The experiments focus on understanding model performance in low-data regimes and cross-county generalization.

**Research Questions**:
- How do TabPFN and XGBoost compare in low-data regimes?
- How well do models trained on pooled data generalize to new counties?
- What is the impact of proper preprocessing to avoid data leakage?
- How do models scale with training data size?

---

## Architecture: Hybrid Preprocessing Pipeline

This repository uses a **two-phase preprocessing approach** to avoid data leakage while maintaining efficiency:

### Phase 1: Data Cleaning (Once)
Heavy cleaning and feature engineering applied to full pooled data:
- Drop null labels, single-value columns, mostly-null columns
- Drop suspicious transactions (lowest ratio sales)
- Drop repeat sales (keep most recent)
- Generate temporal features from `sale_date`
- Label encode categoricals (no one-hot encoding)
- Log transform target

**Output**: Cleaned parquet file (~10-15GB) ready for experiments

### Phase 2: Normalization (Per Train/Test Split)
Statistical transformations fitted on **training data only**, then applied to both train and test:
- Winsorization (clip outliers at train percentiles)
- StandardScaler normalization (using train mean/std)
- Median imputation (using train medians)

**Result**: No data leakage - test statistics never influence preprocessing

---

## Repository Structure

```
tabpfn_data_scarcity/
├── preprocessing/              # Preprocessing pipeline
│   ├── configs/                # Phase 1 configs (data cleaning)
│   │   └── v2_no_onehot.yaml   # Current: label encoding, no one-hot
│   ├── scripts/
│   │   ├── clean_pooled_data.py   # Phase 1 preprocessing
│   │   ├── generate_test_set.py   # Generate test splits
│   │   └── generate_train_set.py  # Generate train splits
│   └── slurm/
│       ├── clean_data.sh          # Phase 1 SLURM job (128GB, 4 hours)
│       ├── generate_test_set.sh   # Test set generation (64GB, 1 hour)
│       └── generate_train_set.sh  # Train set generation (64GB, 1 hour)
│
├── src/                        # Core library (reusable code)
│   ├── models/                 # Model wrappers
│   │   ├── tabpfn_wrapper.py          # Zero-shot TabPFN (ICL)
│   │   ├── tabpfn_finetuning_v2.py    # Global finetuning (current)
│   │   └── xgboost_model.py
│   ├── data/
│   │   ├── loading.py          # CleanedDataLoader (loads parquet)
│   │   ├── preprocessing_utils.py  # Phase 2 preprocessing
│   │   ├── split_strategies.py # Test/train split generation
│   │   ├── filters.py          # County/feature filtering
│   │   └── column_categorizer.py  # Feature categorization
│   ├── evaluation/             # Metrics and result aggregation
│   └── utils/                  # Path utilities, helpers
│
├── experiments/                # Experiment-specific code
│   ├── run_experiment.py       # Main CLI entry point
│   ├── experiment_types/       # Experiment implementations
│   │   ├── base.py             # Base experiment runner (ABC)
│   │   ├── cross_county.py     # Cross-county generalization
│   │   ├── geo_pooling.py      # Geographic neighbor pooling (primary)
│   │   ├── global_finetuning.py # Global TabPFN finetuning
│   │   └── finetuning.py       # Per-county TabPFN finetuning (legacy)
│   ├── configs/
│   │   ├── geo_pooling/        # Geographic pooling configs
│   │   │   ├── nlp/v2_no_onehot/   # NLP cluster configs
│   │   │   │   ├── test_v4_k40_ratio80_droplowest5.yaml          # Baseline
│   │   │   │   ├── test_v4_k40_ratio80_droplowest5_extended_pool.yaml
│   │   │   │   ├── test_v4_k40_ratio80_droplowest5_global_finetuned_external.yaml
│   │   │   │   ├── test_v4_k40_ratio80_droplowest5_global_finetuned_internal.yaml
│   │   │   │   └── test_v4_k40_ratio80_droplowest5_global_finetuned_internal_randsplits/
│   │   │   │       ├── test_v4_rand_s0_...yaml  # Random split seeds 0-4
│   │   │   │       └── ...
│   │   │   └── sherlock/v2_no_onehot/  # Sherlock cluster equivalents
│   │   └── global_finetuning/  # Global finetuning configs
│   │       ├── nlp/v2_no_onehot/
│   │       │   ├── external_15k.yaml
│   │       │   └── internal_15k.yaml
│   │       └── sherlock/v2_no_onehot/
│   └── slurm/                  # SLURM batch job scripts
│       ├── nlp/
│       │   ├── geo_pooling.sh          # Geo pooling array job
│       │   └── global_finetuning.sh    # Global finetuning
│       └── sherlock/
│           ├── geo_pooling.sh
│           └── global_finetuning.sh
│
├── debugging/                  # Diagnostic scripts
│   └── finetuning/
│       ├── README.md                      # Detailed diagnosis writeup
│       ├── instructions.md                # How to run and interpret
│       ├── diag_context_sensitivity.py    # Tier 1: ICL context tests
│       ├── diag_distribution_swap.py      # Tier 2: distribution swap
│       ├── run_diagnostic.sh              # Generic NLP SLURM wrapper
│       └── analysis.ipynb                 # Visualization notebook
│
├── logs/                       # Job logs (gitignored)
│   ├── geo_pooling/
│   └── debugging/finetuning/
│       ├── ft_diagnostic_<jobid>.out
│       └── diag_context_sensitivity_<jobid>.csv
│
├── notebooks/                  # Analysis notebooks
├── data/                       # Metadata (county lists, feature lists)
└── archive/                    # Deprecated code
```

---

## Dataset and Split Versioning

The preprocessed data and test/train splits are versioned independently. The active versions for current experiments are **`v2_no_onehot`** (data) and **`test_v4`** (test set).

### Preprocessed Data Versions

| Version | Description | Path |
|---------|-------------|------|
| `v1_no_onehot` | First version, smaller county set | `/nlp/scr/.../preprocessed/v1_no_onehot/` |
| `v2_no_onehot` | **Current.** Expanded county set, same encoding scheme | `/nlp/scr/.../preprocessed/v2_no_onehot/` |

### Test Set Versions

| Version | # Counties | # Test samples | Split method | Notes |
|---------|-----------|----------------|--------------|-------|
| `test_v1` | 36 | ~1.8K | Temporal 50/50 | Early experiments |
| `test_v4` | **525** | ~25K | **Temporal 80/20** | **Current. Used for all geo-pooling experiments.** |

The 80/20 temporal split means the earliest 80% of transactions per county form the train pool and the most recent 20% are held out as the test set.

### Random Split Variants

In addition to the temporal test_v4 split, random 80/20 splits were generated for evaluating variance across seeds:

| Name | Description |
|------|-------------|
| `test_v4` | Fixed temporal split (default) |
| `test_v4_rand_s0` … `test_v4_rand_s4` | Random 80/20 splits, seeds 0–4 |

Geo-pooling configs for random splits live in `experiments/configs/geo_pooling/*/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_internal_randsplits/`.

### Results Storage

All results are written to `/nlp/scr/salilg/property_tax/results/` (NLP) or `/scratch/users/salilg/property_tax/results/` (Sherlock), organized by experiment type and config name.

---

## Quick Start

### 1. One-Time Setup: Preprocess Data

**Run Phase 1 preprocessing** to create cleaned pooled dataset:

```bash
# Submit to cluster (recommended: 128GB RAM, 4 hours)
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v1_no_onehot.yaml

# Monitor progress
tail -f preprocessing/logs/clean_*.out

# Output: /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/
#   ├── data.parquet            (~10-15GB compressed)
#   ├── metadata.json           (statistics, column info)
#   ├── config.yaml             (config used)
#   └── preprocessing_log.txt   (detailed log)
```

**What Phase 1 does**:
- Loads all 87 high-coverage counties (~21M rows → ~15M after cleaning)
- Applies data cleaning (nulls, duplicates, outliers)
- Generates temporal features (year, month, day, etc.)
- Label encodes categoricals (not one-hot)
- Log transforms target
- Saves to single parquet file

**This only needs to be done once**. All experiments load from this cleaned data.

### 2. Generate Test/Train Splits (Pre-Generated Approach)

**Why pre-generate splits?** For reproducibility, comparability across experiments, and auditability. Generate once, use many times.

#### Step 2a: Generate Test Set

```bash
# Generate test set (size-stratified counties with temporal split)
sbatch preprocessing/slurm/generate_test_set.sh \
    experiments/configs/test_sets/test_v1.yaml \
    experiments/splits/test_v1/

# Monitor
tail -f preprocessing/logs/gen_test_*.out

# Output: experiments/splits/test_v1/
#   ├── test_indices.npy           (test set indices)
#   ├── train_pool_indices.npy     (available training indices)
#   ├── test_set_config.json       (config used)
#   ├── county_info.json           (per-county statistics)
#   ├── size_buckets.json          (size stratification)
#   └── summary_report.txt         (human-readable summary)
```

**What test set generation does**:
- Selects counties stratified by size (tiny, small, medium, large, xlarge) — count depends on config (e.g. test_v1 = 36 counties, test_v4 = 525 counties)
- Within each test county, splits by date (temporal split fraction set in config)
- Saves indices to disk for reproducibility

#### Step 2b: Generate Training Sets

```bash
# Generate training set (example: train_v2 = mixed 50% history, 50% external)
sbatch preprocessing/slurm/generate_train_set.sh train_v2 experiments/splits/test_v1/

# Monitor
tail -f preprocessing/logs/gen_train_*.out

# Output: experiments/splits/test_v1/train_v2/
#   ├── train_indices.npy          (training set indices)
#   ├── train_set_config.json      (config used)
#   ├── source_breakdown.json      (where samples came from)
#   ├── county_distribution.json   (samples per county)
#   └── summary_report.txt         (human-readable summary)
```

**Available training strategies** (see `experiments/configs/train_sets/README.md`):
- **train_v1**: Test county history only (bottom 50% temporal)
- **train_v2**: Mixed - 50% test history, 50% external counties (10K cap)
- **train_v3**: External only - pure cross-county (10K cap)
- **train_v4**: Stratified external - maximize diversity (10K cap)
- **train_v5**: Large-scale - all available data (for XGBoost/fine-tuning)

**Generate multiple training sets** for comparison:
```bash
# Generate all training strategies for test_v1
for version in train_v1 train_v2 train_v3 train_v4 train_v5; do
    sbatch preprocessing/slurm/generate_train_set.sh $version experiments/splits/test_v1/
done
```

### 3. Run Experiments

Once splits are pre-generated, run experiments using the saved splits:

```bash
# Cross-county experiment using pre-generated splits
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml

# Monitor
squeue -u $USER
tail -f experiments/logs/cross_county_*.out
```

**What experiments do**:
- Load cleaned parquet (fast, cached in memory)
- Load pre-generated test/train splits from disk
- Apply Phase 2 preprocessing (fit on train only)
- Train models, evaluate, save results

**Compare different training strategies**:
```bash
# Run experiments with different training approaches
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v1.yaml
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v3.yaml
```

All experiments use the **same test set** (test_v1), ensuring fair comparison across training strategies

---

## Data Pipeline

### Data Source
- **Location**: `/scratch/users/salilg/property_tax/county_csvs/`
- **Format**: Individual CSV files per county (e.g., `fips_17031.csv`)
- **Target Variable**: `SALE_AMOUNT` (property sale price)
- **Metadata**: County info at `data/county_row_counts.csv`

### Feature Categories
- **Property Characteristics**: Beds, baths, square footage, lot size, year built, etc.
- **Census Block Group**: Demographic/economic variables (optional, disabled by default)
- **Assessed Values**: Property tax assessments (optional, disabled by default)
- **Geographic**: Latitude, longitude (optional, disabled by default)
- **Temporal**: Sale year, month, day, day of week, days since 2000 (generated from `sale_date`)

### Preprocessing Configuration

**Phase 1 config** (`preprocessing/configs/v1_no_onehot.yaml`):
```yaml
# Feature selection
features:
  property_chars: true      # Property characteristics
  census_bg: false          # Census features (disabled)
  assessed_value: false     # Assessed values (disabled)
  geographic: false         # Lat/lon (disabled)
  temporal: true            # Generate temporal features

# Phase 1 steps (applied to full pooled data)
phase1_steps:
  drop_null_labels: true
  drop_single_value_cols: true
  drop_mostly_null_cols: true
  share_non_null: 0.5
  drop_lowest_ratios: true      # Remove suspicious transactions
  drop_repeat_sales: true       # Keep only most recent sale
  generate_temporal_features: true
  categorical_handling: "label_encode"  # Not one-hot
  log_transform_target: true

# Phase 2 steps (documented but applied per-experiment)
phase2_steps:
  winsorize: true
  winsorize_percentile: 1
  normalize_continuous: true
  impute_method: "median"
```

**Experiment config** (`experiments/configs/cross_county/cross_county_v2.yaml`):
```yaml
# Point to preprocessed data
data:
  cleaned_data_path: "/scratch/.../preprocessed/cleaned_datasets/v1_no_onehot/"
  target_column: "SALE_AMOUNT"

# Phase 2 steps (fit on train, apply to train and test)
preprocessing:
  phase2_steps:
    winsorize: true
    winsorize_percentile: 1
    normalize_continuous: true
    impute_method: "median"
```

---

## Experiment Types

### Cross-County Generalization (Primary Focus)

Test how well models trained on different data sources generalize to held-out test counties.

**New Workflow** (with pre-generated splits):

**Config**: `experiments/configs/cross_county/test_v1_train_v2.yaml`

**Usage**:
```bash
# First: Generate test and train splits (one-time setup)
sbatch preprocessing/slurm/generate_test_set.sh experiments/configs/test_sets/test_v1.yaml experiments/splits/test_v1/
sbatch preprocessing/slurm/generate_train_set.sh train_v2 experiments/splits/test_v1/

# Then: Run experiment using pre-generated splits
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml
```

**How it works**:
1. Loads cleaned parquet data (all counties)
2. Loads pre-generated test set (36 size-stratified counties)
3. Loads pre-generated training set (10K samples with specific strategy)
4. Applies Phase 2 preprocessing (fit on train only)
5. Trains TabPFN and XGBoost on training data
6. Evaluates on each test county
7. Saves per-county results

**Output**: `/scratch/users/salilg/property_tax/results/cross_county/test_v1_train_v2/`

**Key Features**:
- **Reproducible**: Same test/train splits across runs
- **Comparable**: Different training strategies use identical test sets
- **Auditable**: Splits saved to disk with metadata and summaries
- **No data leakage**: Phase 2 preprocessing fit on train only
- **Fast**: Loads from parquet, pre-computed splits
- **Size-stratified**: Test counties span 5 size buckets

**Research Questions**:
- **train_v1** (test history only): Can we predict recent sales using historical data from same county?
- **train_v2** (mixed): Does combining county history with external data help?
- **train_v3** (external only): Pure cross-county generalization - how well do other counties' data transfer?
- **train_v4** (stratified external): Does maximizing county diversity improve generalization?
- **train_v5** (large-scale): How much does XGBoost benefit from unlimited training data?

### Per-County Scaling Experiment

Build per-county learning curves by training separate models on each county's own historical data with varying training set sizes. Targets tiny (2-100 rows) and small (100-500 rows) counties from test_v1.

**Config**: `experiments/configs/per_county_scaling/tiny_small.yaml`

**Usage**:
```bash
# First: Generate county FIPS list (one-time)
python -c "
from src.data.split_strategies import load_test_set_result
result = load_test_set_result('/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/')
fips_list = []
for bucket in ['tiny', 'small']:
    fips_list.extend(result.size_buckets.get(bucket, []))
fips_list.sort()
with open('experiments/slurm/per_county_fips_list.txt', 'w') as f:
    for fips in fips_list:
        f.write(f'{fips}\n')
print(f'Wrote {len(fips_list)} counties to per_county_fips_list.txt')
"

# Then: Submit SLURM array job (200 tasks, one per county)
sbatch experiments/slurm/per_county_scaling.sh experiments/configs/per_county_scaling/tiny_small.yaml

# Or test a single county locally:
python experiments/run_experiment.py \
    --experiment_type per_county_scaling \
    --config experiments/configs/per_county_scaling/tiny_small.yaml \
    --county_fips 31007
```

**How it works**:
1. Each SLURM array task processes one county
2. For each county, sweeps over train_sizes × seeds × models
   - Tiny counties: ~6 sizes × 3 seeds × 2 models = 36 trainings (~20 min)
   - Small counties: ~11 sizes × 3 seeds × 2 models = 66 trainings (~45 min)
3. For each combo: samples training points, applies Phase 2 preprocessing (fit fresh), trains model
4. Adaptive XGBoost tuning: reduces CV folds to 2 when train_size < 30
5. Checkpoints every 50 combos, resumes automatically on restart
6. Saves per-county results to separate subdirectories

**Output**: `/scratch/users/salilg/property_tax/results/per_county_scaling/tiny_small/`
```
results/per_county_scaling/tiny_small/
├── county_31007/
│   ├── results.csv           # All train_size × seed × model rows
│   ├── results_checkpoint.csv
│   └── completed_keys.pkl
├── county_31009/
│   └── ...
└── experiment.log
```

**Key Features**:
- **Per-county models**: Each county gets its own model trained only on its historical data
- **Learning curves**: Vary training size from 5 to 250 samples to see how performance scales
- **Multiple seeds**: 3 random seeds per train_size for statistical robustness
- **Checkpointing**: Resume automatically if interrupted
- **Memory efficient**: Loads only needed rows from data.parquet
- **Parallelizable**: 200 SLURM array tasks run in parallel

**Research Questions**:
- How much historical data does a county need for good predictions?
- Do tiny counties benefit from even 5-10 historical samples?
- At what training size do TabPFN and XGBoost converge?
- Does the answer vary by county size/characteristics?

**Smoke Test** (runs on interactive nodes with 4-8GB memory):
```bash
# Run end-to-end test with minimal fake data (~2-3 minutes)
python experiments/scripts/smoke_test_per_county_scaling.py
```

This creates 2 fake counties (50 rows each), runs the full pipeline for 1 county with 2 train sizes, and verifies correctness. Use this to test the code before submitting large SLURM jobs.

### Geographic Pooling (Primary Focus)

Per-county prediction using geographically pooled training data from nearby counties. Each test county's model is trained on its own historical data plus data sampled from neighboring counties, weighted by geographic proximity.

**How it works**:
1. Loads test_v4 counties and their train/test splits
2. For each test county, finds the k nearest neighbor counties using geographic centroids
3. Builds a training pool: own historical data + neighbor county data (closest counties first, capped by a budget ratio)
4. Applies Phase 2 preprocessing and ratio filtering
5. Trains TabPFN (zero-shot ICL) and XGBoost on the pooled data
6. Evaluates on the county's held-out test set
7. Saves per-county results + neighbor usage diagnostics

**Usage**:
```bash
# Array job: splits test counties across 4 tasks
sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
  experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5.yaml
```

**Key Config Parameters** (`geo_pooling` section):
```yaml
geo_pooling:
  centroids_csv: "data/us_county_latlng.csv"
  max_k_neighbors: 40            # Max neighbors to consider
  neighbor_budget_ratio: 0.8     # Fraction of total budget for neighbor data
  max_total_training_size: 10000  # TabPFN context limit
```

#### Extended Neighbor Pool

By default, neighbor candidates are restricted to test_v4 counties (525 of 2667). Enable `extend_neighbor_pool_beyond_testv4` to draw neighbor data from ALL ~2667 counties in the dataset.

```yaml
geo_pooling:
  extend_neighbor_pool_beyond_testv4: true  # Use all counties as neighbor candidates
```

**How it works**: Reads the fips column from the full `data.parquet` to discover all counties, identifies which are outside test_v4, loads their data in bulk, applies ratio filtering, and adds them to the neighbor candidate pool. The neighbor selection still uses geographic distance — closest counties are prioritized first.

**Memory**: The full parquet read peaks at ~2.5GB. SLURM nodes with 32GB+ handle this without issue.

#### Diversified Neighbor Sampling

When the nearest neighbor county alone has enough data to fill the entire training budget, this can lead to a homogeneous training set dominated by one county. Enable `diversify_neighbors` to split the budget equally among the N nearest neighbors instead.

```yaml
geo_pooling:
  diversify_neighbors: true   # Split budget among N nearest when one could fill it all
  diversify_n: 3              # Number of neighbors to spread across (default: 3)
```

**When it triggers**: Only when (1) diversification is enabled, (2) at least `diversify_n` neighbors are available, and (3) the closest neighbor county has enough data to fill the entire budget by itself. Otherwise falls through to the standard greedy (closest-first) allocation.

**Example config** (extended pool + diversification):
```bash
sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
  experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_extended_pool.yaml
```

#### Using a Globally Finetuned TabPFN

Instead of zero-shot TabPFN, you can use a pre-finetuned model checkpoint as a drop-in replacement. The finetuned model still uses in-context learning with county-specific training data — it just starts from a better-adapted base model.

```yaml
models:
  - name: "tabpfn_global_finetuned"
    enabled: true
    checkpoint_dir: "/nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/external_15k/"
```

**Prerequisites**: Run the global finetuning experiment first (see below) to create the checkpoint.

**Example** (full workflow, fixed test split):
```bash
# Step 1: Global finetuning (creates checkpoint)
sbatch experiments/slurm/nlp/global_finetuning.sh \
  experiments/configs/global_finetuning/nlp/v2_no_onehot/external_15k.yaml

# Step 2: Geo pooling with the finetuned model
sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
  experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_external.yaml
```

**Example** (internal variant across all 5 random splits, NLP):
```bash
# Step 1: Global finetuning — internal variant (train pool of test counties)
sbatch experiments/slurm/nlp/global_finetuning.sh \
  experiments/configs/global_finetuning/nlp/v2_no_onehot/internal_15k.yaml

# Step 2: Geo pooling across random splits s0-s4
for s in 0 1 2 3 4; do
  sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
    experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_internal_randsplits/test_v4_rand_s${s}_k40_ratio80_droplowest5_global_finetuned_internal.yaml
done
```

**Example** (internal variant across all 5 random splits, Sherlock):
```bash
# Step 1: Global finetuning — internal variant
sbatch experiments/slurm/sherlock/global_finetuning.sh \
  experiments/configs/global_finetuning/sherlock/v2_no_onehot/internal_15k.yaml

# Step 2: Geo pooling across random splits s0-s4
for s in 0 1 2 3 4; do
  sbatch --array=0-3 experiments/slurm/sherlock/geo_pooling.sh \
    experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_internal_randsplits/test_v4_rand_s${s}_k40_ratio80_droplowest5_global_finetuned_internal.yaml
done
```

**Output** (each chunk gets its own subdirectory):
```
results/geo_pooling/v2_no_onehot/test_v4_geo_k40_ratio80_droplowest5/
├── chunk_0/
│   ├── results.csv              # Per-county metrics for all models
│   ├── results_checkpoint.csv   # Checkpoint for resume
│   ├── completed_keys.pkl       # Set of (fips, model) keys already done
│   └── neighbor_usage.parquet   # Which neighbors contributed data for each county
├── chunk_1/ ...
├── chunk_2/ ...
├── chunk_3/ ...
└── experiment.log
```

**Re-running an experiment**: By default, re-submitting a job overwrites previous results (no auto-resume). To resume a partially completed run instead (e.g. a job that was killed halfway), add to the config:

```yaml
checkpointing:
  resume: true
```

---

### Global Finetuning

Finetune TabPFN once on a large pooled dataset (10-15K samples from many counties), save the checkpoint, then use it as a drop-in ICL replacement in geo pooling experiments.

**Two variants**:
- **(a) External** (`variant: "external"`): Finetune on data from non-test_v4 counties. No test data leakage since these counties are completely separate from the test set.
- **(b) Internal** (`variant: "internal"`): Finetune on the train pool indices of test_v4 counties (historical county data). Uses the same counties but only their training data.

**Usage**:
```bash
# External variant: finetune on non-test counties
sbatch experiments/slurm/nlp/global_finetuning.sh \
  experiments/configs/global_finetuning/nlp/v2_no_onehot/external_15k.yaml

# Internal variant: finetune on test county train pools
sbatch experiments/slurm/nlp/global_finetuning.sh \
  experiments/configs/global_finetuning/nlp/v2_no_onehot/internal_15k.yaml
```

**Key Config Parameters**:
```yaml
global_finetuning:
  variant: "external"          # "external" or "internal"
  n_samples: 15000             # Number of rows to sample for finetuning
  sampling_strategy: "uniform" # "uniform" or "stratified_by_county"

finetuning:
  learning_rate: 1e-4
  max_epochs: 100
  patience: 16                 # Early stopping epochs
  seq_len_pred: 1024           # Query samples per step
  max_context_size: 10000      # Context cap for memory
  val_fraction: 0.2            # Internal validation split
```

**Output**:
```
results/global_finetuning/v2_no_onehot/external_15k/
├── model.pt                 # Finetuned model weights
├── metadata.json            # Config, column info, target stats
├── transforms.pkl           # Target/prediction transforms
├── training_history.json    # Loss curves for plotting
└── results.csv              # Summary metrics
```

**Preprocessing consistency for `training_mode: "per_county"`**: During finetuning, each training step normalizes the county's numerical features by the context mean/std (same as StandardScaler, but computed per county per step). This exactly matches the per-county Phase 2 StandardScaler applied at inference time in geo_pooling. The global Phase 2 preprocessing applied before `fit()` applies winsorization and imputation but skips StandardScaler (`normalize_continuous: false` is set internally) to avoid double-scaling.

**Plotting loss curves** from saved training history:
```python
import json
import matplotlib.pyplot as plt

with open('.../training_history.json') as f:
    history = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_losses'], label='Train loss')
# Note: val_losses is always empty; val R² is in val_metrics instead
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

ax2.plot([m.get('r2', None) for m in history['val_metrics']], label='Val R²')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R²'); ax2.legend()
plt.tight_layout(); plt.show()
```

**Implementation files**:
- Experiment: `experiments/experiment_types/global_finetuning.py`
- Model save/load: `src/models/tabpfn_finetuning_v2.py` (`save_to_disk()`, `load_from_disk()`)
- SLURM scripts: `experiments/slurm/nlp/global_finetuning.sh` / `experiments/slurm/sherlock/global_finetuning.sh`
- Configs (NLP): `experiments/configs/global_finetuning/nlp/v2_no_onehot/`
- Configs (Sherlock): `experiments/configs/global_finetuning/sherlock/v2_no_onehot/`

### Hyperparameter Sweep (LoRA + LR + Epoch Size)

Run a grid search over finetuning hyperparameters across one or more data splits, with geo pooling jobs automatically queued as SLURM dependencies.

**Grid** (24 combinations):
- `learning_rate`: 1e-5, 5e-5, 1e-4, 5e-4
- `lora_rank`: 4, 8, 16
- `epoch_size`: 50, 100

All runs use LoRA finetuning and the full ~99K training pool.

**Splits**:
- `--temporal`: temporal 80/20 split (`test_v4/`)
- `--seeds N ...`: random 80/20 splits (`test_v4_rand_sN/`) — requires those test sets to exist on scratch

**Prerequisite**: random split test sets (`test_v4_rand_s0/` … `test_v4_rand_s4/`) must exist on scratch. Seeds 0–4 were already generated in a previous session and should be present at `/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_sN/`. If any are missing, regenerate them:
```bash
for S in 0 1 2 3; do
  python experiments/scripts/generate_test_set.py \
    --config experiments/configs/test_sets/test_v4_rand.yaml \
    --data_path /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/data.parquet \
    --output_dir /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s${S}/ \
    --split_seed $S
done
```

**Target normalization fix (2026-03)**: The training loop clamps normalized targets
`Y_ctx_norm` and `Y_qry_norm` to `[-10, 10]`, matching existing feature clamping.
Without this fix, a random context split that draws 2 samples with nearly-identical
log prices produces `y_ctx_std ≈ 1e-8`, causing `Y_qry_norm` to blow up to ±millions
and the step loss to reach ~10¹¹. The spike diagnostic confirmed this on fips 31007
(county_size=6, n_ctx=2, y_ctx_std=1e-8). The fix is in
`src/models/tabpfn_finetuning_v2.py` (`_fit_per_county`, per-county y normalization
block). All previous sweep results pre-date this fix.

**Before running the full sweep, test with 1–2 configs**:
```bash
# Test 1: lora8, lr=1e-4, ep100, seed 0 (the config that produced catastrophic spikes)
sbatch experiments/slurm/sherlock/global_finetuning.sh \
  experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/lora8_lr1e-4_ep100_s0.yaml

# Test 2: lora8, lr=5e-4, ep100, seed 0 (highest LR — most likely to reveal instability)
sbatch experiments/slurm/sherlock/global_finetuning.sh \
  experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/lora8_lr5e-4_ep100_s0.yaml
```
Check the output logs in `logs/global_finetuning/sherlock/` — specifically confirm that
`train_loss` does not contain epoch-mean values above ~10 in the history JSON.

**Spike diagnostic** (for future debugging — produces `spike_diagnostics.jsonl` +
`zeroshot_per_county.json`):
```bash
# Run spike diagnostic on Sherlock (captures detailed per-step info for any loss > 10)
sbatch debugging/finetuning/run_diagnostic_sherlock.sh \
  debugging/finetuning/diag_training_spikes.py \
  --lora_rank 8 --learning_rate 1e-4 --epoch_size 100 --max_epochs 50

# Adjust spike threshold to catch smaller instabilities
sbatch debugging/finetuning/run_diagnostic_sherlock.sh \
  debugging/finetuning/diag_training_spikes.py \
  --lora_rank 8 --learning_rate 5e-4 --epoch_size 100 --max_epochs 50 \
  --spike_threshold 5
```
Output is saved to `logs/debugging/finetuning/spike_diag_<tag>_<jobid>/`.

**Usage**:
```bash
# Dry run — print all commands without submitting
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --dry-run

# Full sweep: temporal split + 4 random seeds
# (120 FT jobs + 240 geo pooling jobs; geo pooling queued with --dependency=afterok)
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3

# Temporal split only (24 FT + 48 geo pooling)
python experiments/scripts/sweep.py --temporal

# Finetuning only (skip geo pooling)
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --ft-only

# Geo pooling only — re-run inference when finetuning checkpoints already exist
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --geo-only
```

**Run naming**:
- Temporal split: `lora8_lr1e-4_ep100`
- Random seed N: `lora8_lr1e-4_ep100_sN`

**Generated configs** (all under `sweep/` subdirs, isolated from existing configs):
```
experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/<name>.yaml
experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/<name>_nopooling.yaml
experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/<name>_ratio80.yaml
```

**Results**:
```
/scratch/users/salilg/property_tax/results/global_finetuning/v2_no_onehot/sweep/<name>/
/scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/sweep/<name>_nopooling/
/scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/sweep/<name>_ratio80/
```

Each geo pooling run evaluates only `tabpfn_global_finetuned` (zero-shot models disabled).

**Implementation**: `experiments/scripts/sweep.py`

---

### Legacy Experiment (Deprecated)

**Config**: `experiments/configs/cross_county/cross_county_v2.yaml`

This config generates splits on-the-fly (all counties as test, all others as train). While supported for backward compatibility, the pre-generated splits approach (above) is strongly recommended for new experiments.

### Other Experiment Types

The repository includes other experiment types (within-county CV, data scaling, fine-tuning) but these have not yet been migrated to the new preprocessing architecture.

---

## Configuration Files

All experiments are configured via YAML files.

### Preprocessing Config (Phase 1)

Located in `preprocessing/configs/`, these define how raw data is cleaned:

```yaml
version: "v1_no_onehot"
description: "Cleaned data with label encoding, no one-hot"

data:
  county_csvs_dir: "/scratch/.../county_csvs/"
  target_column: "SALE_AMOUNT"

data_filtering:
  counties:
    enabled: true
    file: "data/county_metadata/high_coverage_87_county_list.csv"
  features:
    enabled: true
    file: "data/feature_lists/high_coverage__87_county_features.csv"

features:
  property_chars: true
  temporal: true
  # ... other categories disabled

phase1_steps:
  # ... cleaning steps (see above)

output:
  base_dir: "/scratch/.../preprocessed/cleaned_datasets/"
  format: "parquet"
  compression: "snappy"
```

### Experiment Config (Phase 2)

Located in `experiments/configs/`, these define experiment parameters:

```yaml
experiment:
  type: "cross_county"
  name: "cross_county_v2"

data:
  cleaned_data_path: "/scratch/.../cleaned_datasets/v1_no_onehot/"

preprocessing:
  phase2_steps:
    winsorize: true
    normalize_continuous: true
    impute_method: "median"

iterations: 10  # Random iterations per target county

models:
  - name: "tabpfn"
    enabled: true
  - name: "xgboost"
    enabled: true

xgboost:
  optuna_trials: 50
  optuna_cv_folds: 3
  use_gpu: true

output:
  results_dir: "/scratch/.../results/cross_county/v2/"
```

---

## Results and Analysis

### Result Structure

Results are saved to the directory specified in experiment config:

```
results/cross_county/v2/
├── results.csv         # Main results (all counties, all iterations)
└── experiment.log      # Detailed log
```

### Result Columns

- `model`: Model name (tabpfn, xgboost)
- `target_fips`: County used for testing
- `iteration`: Random iteration number (0-9)
- `n_train_counties`: Number of training counties
- `train_size`, `test_size`: Sample counts
- `n_features`: Number of features after preprocessing
- `r2`, `mae`, `rmse`, `mse`: Evaluation metrics
- `fit_time`, `pred_time`: Training and inference time (seconds)
- `status`: "success" or error message

**Note**: Metrics are computed on log-transformed target, then converted back to original scale.

---

## SLURM Job Management

### Resource Requirements

| Job Type | Memory | CPUs | GPU | Time |
|----------|--------|------|-----|------|
| Preprocessing (Phase 1) | 128GB | 8 | No | 4 hours |
| Generate test set | 64GB | 4 | No | 1 hour |
| Generate train set | 64GB | 4 | No | 1 hour |
| Cross-county experiment | 64GB | 4 | 1x | 12 hours |
| Geo pooling (per chunk) | 64GB | 4 | 1x | 12 hours |
| Global finetuning | 64GB | 4 | 1x | 4 hours |

### Common Commands

```bash
# Submit preprocessing (Phase 1)
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v1_no_onehot.yaml

# Generate splits
sbatch preprocessing/slurm/generate_test_set.sh experiments/configs/test_sets/test_v1.yaml experiments/splits/test_v1/
sbatch preprocessing/slurm/generate_train_set.sh train_v2 experiments/splits/test_v1/

# Submit experiment
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml

# Monitor jobs
squeue -u $USER

# View logs
tail -f preprocessing/logs/clean_*.out
tail -f preprocessing/logs/gen_test_*.out
tail -f preprocessing/logs/gen_train_*.out
tail -f experiments/logs/cross_county_*.out

# Cancel jobs
scancel <job_id>
```

### Log Files

- Preprocessing logs: `preprocessing/logs/clean_<jobid>.{out,err}`
- Test set generation logs: `preprocessing/logs/gen_test_<jobid>.{out,err}`
- Train set generation logs: `preprocessing/logs/gen_train_<jobid>.{out,err}`
- Experiment logs: `experiments/logs/cross_county_<jobid>.{out,err}`

---

## Models

### TabPFN (Tabular Prior-Fitted Network)

- **Type**: Transformer-based in-context learning
- **Strengths**: Zero hyperparameter tuning, fast, built-in uncertainty
- **Limits**: Max 10k samples, 100 features
- **Implementation**: `src/models/tabpfn_model.py`

### XGBoost

- **Type**: Gradient boosted decision trees
- **Strengths**: Strong baseline, handles large data
- **Configuration**: Hyperparameter tuning via Optuna (50 trials, 3-fold CV)
- **Implementation**: `src/models/xgboost_model.py`

---

## TabPFN Fine-Tuning

### v2: Yandex-Style Direct Fine-Tuning (Current - Feb 2026)

**Status**: ✅ **Recommended** - Faithful implementation of the [Yandex tabpfn-finetuning paper](https://arxiv.org/abs/2506.08982)

The v2 implementation loads the TabPFN v2 checkpoint directly and runs a standard PyTorch training loop, keeping optimizer parameter references stable so parameters actually update.

**What Changed from v1**:
- **Load checkpoint directly**: Bypasses the high-level `TabPFNRegressor` API that was breaking optimizer references
- **Stable optimizer**: Parameters stay the same objects throughout training, so optimizer momentum accumulates correctly
- **Context capping**: `max_context_size` parameter caps how many training samples are used as context per step (default: null = all). Required for large training sets to avoid OOM on 80GB A100.
- **Query sampling**: Samples `seq_len_pred` query indices per step (default: 128 for geo pooling)
- **Bar distribution loss**: Uses 5000-bin bar distribution for regression predictions
- **Verified parameters update**: Non-zero parameter changes after first epoch
- **Auto val split** (Mar 2026): If no `X_val`/`y_val` passed to `fit()`, an internal val split is created using `val_fraction` and `random_state` from config — reproducible across runs, used in geo pooling mode
- **Auto continuous_cols** (Mar 2026): If `continuous_cols` not passed, detected automatically from dtype (numeric with >2 unique values)
- **Geo pooling integration** (Mar 2026): `tabpfn_finetuned` added as a model option in geo pooling experiments, enabling per-county finetuning on the county's geo-pooled training set

**Architecture** (from Yandex paper):
```
Training Loop:
  For each epoch (epoch_size steps):
    Sample seq_len_pred query indices
    Context = random sample of min(max_context_size, remaining) training samples
    Forward: embed features → embed targets → transformer → decoder
    Loss: bar distribution (5000 bins) on query samples
    Backward → gradient clip → optimizer step
  Validation: use full train as context, predict val, compute R²
  Early stopping: patience epochs
```

**Key Config Parameters**:
```yaml
finetuning:
  learning_rate: 1e-4
  epoch_size: 10              # Steps per epoch
  seq_len_pred: 128           # Query samples per step (128 for geo pooling)
  max_context_size: null      # null = all training samples (fine for small geo pooling counties)
  batch_size: 1               # Independent ICL tasks
  patience: 8                 # Early stopping (shorter for per-county runs)
  finetune_mode: "full"       # Full model finetuning
  gradient_clip: 1.0
  use_amp: true               # bfloat16 mixed precision
  val_fraction: 0.2           # Used for internal val split when no X_val provided
  min_train_size: 20          # (geo pooling only) skip counties smaller than this
```

**Usage — Standalone finetuning**:
```bash
sbatch experiments/slurm/sherlock/finetune_tabpfn_v2.sh \
  experiments/configs/finetuning/sherlock/test_v4_train_v1.yaml
```

**Usage — Geo pooling + per-county finetuning (recommended)**:
```bash
# Array job across 4 chunks of counties
sbatch --array=0-3 experiments/slurm/sherlock/geo_pooling_finetune.sh \
  experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5.yaml
```

**What to Expect in Logs**:
- Zero-shot val R² (before training)
- Per-epoch: train_loss, val_r2, learning_rate, time
- "New best epoch!" when val R² improves
- "Params with non-zero grad: X/Y" after epoch 1 (verifies parameters are updating)
- Final: "Restored best model from epoch N"

**Implementation Files**:
- Model: `src/models/tabpfn_finetuning_v2.py`
- Architecture components: `src/models/tabpfn_lib/` (copied from Yandex repo)
- Standalone experiment runner: `experiments/experiment_types/finetuning.py`
- Geo pooling integration: `experiments/experiment_types/geo_pooling.py` (`create_model('tabpfn_finetuned')`)
- Geo pooling config (Sherlock): `experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5.yaml`
- Geo pooling SLURM (Sherlock): `experiments/slurm/sherlock/geo_pooling_finetune.sh`

**Known Limitations (from experiments — see `experiments/docs/FINETUNING_EXPERIMENTS_LOG.md`)**:
- On large heterogeneous training sets (e.g. 8000 samples from 66 mixed counties), finetuning gives marginal gains with significant overfitting. The geo pooling setting is better suited: small, coherent per-county training sets with a naturally representative val split.
- `max_context_size` is required when train_size > ~2000 on 80GB A100, because backprop stores all intermediate activations (unlike inference).

---

### v1: High-Level API Fine-Tuning (Legacy - Broken)

**Status**: ⚠️ **Deprecated** - Critical optimizer bug causes parameters to never update

The v1 implementation used TabPFN's high-level `TabPFNRegressor` API (`fit_from_preprocessed()`, `forward()`) which recreated internal model objects on each call, breaking optimizer parameter references.

**Root Cause** (`src/models/tabpfn_finetuning.py`):
- Called `fit_from_preprocessed()` every batch
- This recreated `model_` object with NEW parameter tensors
- Optimizer still held references to OLD parameters
- Result: optimizer updates ghost parameters, model weights never change

**Evidence of Bug**:
```
Avg parameter change: 0.00e+00, Params that changed: 0/81
val_loss: 0.356194 (identical across all 11 epochs)
Optimizer state: INVALID
```

**To Use v1** (not recommended):
```yaml
finetuning:
  implementation: "v1"  # Use legacy broken approach
```

This is only kept for backward compatibility and comparison. **Use v2 for all new experiments.**

---

## Troubleshooting

### Preprocessing Fails

```bash
# Check log
cat preprocessing/logs/clean_<jobid>.err

# Verify paths in config
vim preprocessing/configs/v1_no_onehot.yaml

# Check county/feature filter files exist
ls data/county_metadata/high_coverage_87_county_list.csv
ls data/feature_lists/high_coverage__87_county_features.csv
```

### Experiment Fails

```bash
# Check cleaned data exists
ls /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet

# Verify config points to correct path
vim experiments/configs/cross_county/cross_county_v2.yaml

# Check experiment log
cat experiments/logs/cross_county_<jobid>.err
```

### Out of Memory

**Preprocessing**: Request more memory in `preprocessing/slurm/clean_data.sh`:
```bash
#SBATCH --mem=256G  # Increase from 128G
```

**Experiments**: Experiments cache the full parquet in memory. If memory issues occur:
- Reduce number of counties
- Sample data in experiment config

---

## Advanced: Creating New Preprocessed Datasets

To create a new preprocessed dataset with different settings:

1. **Copy and modify config**:
```bash
cp preprocessing/configs/v1_no_onehot.yaml preprocessing/configs/v2_custom.yaml
# Edit v2_custom.yaml to change settings
```

2. **Run preprocessing**:
```bash
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v2_custom.yaml
```

3. **Update experiment config** to use new dataset:
```yaml
data:
  cleaned_data_path: "/scratch/.../cleaned_datasets/v2_custom/"
```

Common variations:
- **One-hot encoding**: Set `categorical_handling: "one_hot"` (increases features significantly)
- **Different features**: Enable/disable census, geographic, assessed value features
- **Different counties**: Modify `data_filtering.counties.file`
- **Different threshold**: Adjust `share_non_null` for column dropping

---

## Tips

1. **Three-step workflow**: Phase 1 (preprocess) → Generate splits → Run experiments. Each step only needs to be done once.
2. **Pre-generate splits for reproducibility**: Always use pre-generated splits for production experiments. Same test set ensures fair comparison across training strategies.
3. **Check summary reports**: After generating splits, review `summary_report.txt` files to verify splits are reasonable.
4. **Test multiple training strategies**: Generate all train_v1 through train_v5 for your test set, then compare results.
5. **Check metadata.json**: Inspect preprocessing output to understand what was done.
6. **Start with fewer counties**: For test set configs, reduce n_counties per bucket to test workflow quickly.
7. **Monitor Phase 2 time**: If experiments are slow, Phase 2 preprocessing may be the bottleneck.
8. **Parquet is fast**: Loading 15M rows from parquet takes ~10 seconds vs minutes from CSV.

---

## Understanding Prediction File Indices

Prediction files (`.parquet` files with individual predictions) use a `test_index` column to identify which data.parquet row each prediction corresponds to. However, **different scripts use different index schemes**:

### Baseline Predictions
**File**: `test_v1/baseline_predictions.parquet`
**Generated by**: `experiments/scripts/generate_baseline_results.py`

- `test_index` = **original data.parquet row indices** (from `test_indices.npy`)
- Range: 35862 to 17904827 (sparse)
- Directly maps to rows in the full data.parquet file

### Finetuning Predictions
**Files**: `results/finetuning/*/predictions_*.parquet`
**Generated by**: `experiments/experiment_types/finetuning.py`

- `test_index` = **iloc positions within a subset DataFrame**
- Range: 0 to ~5.5M (not all values used)
- The finetuning code loads only needed rows (test + train_pool + train) into a subset, then uses iloc positions within that subset
- These are NOT direct data.parquet row indices

### Converting Finetuning test_index to data.parquet Row Index

To join finetuning predictions with baseline predictions (or map to FIPS), convert the finetuning `test_index` to original data.parquet row indices:

```python
import numpy as np

# Load the indices used to create the subset DataFrame
test_indices = np.load('.../test_v1/test_indices.npy')
train_pool_indices = np.load('.../test_v1/train_pool_indices.npy')
train_indices = np.load('.../test_v1/train_v6/train_indices.npy')  # Use appropriate train version

# Reconstruct the unique_indices array (sorted, deduplicated)
unique_indices = np.unique(np.concatenate([
    test_indices,
    train_pool_indices,
    train_indices
]))

# Convert finetuning test_index -> data.parquet row index
data_row_index = unique_indices[finetuning_test_index]
```

### Joining Prediction Files at Row Level

To join baseline and finetuning predictions for the same test samples:

```python
import pandas as pd

# Load predictions
xgb = pd.read_parquet('.../predictions_xgboost.parquet')
baseline = pd.read_parquet('.../baseline_predictions.parquet')

# Convert both to data.parquet row indices
xgb['data_row_index'] = unique_indices[xgb['test_index'].values]  # Convert finetuning indices
baseline['data_row_index'] = baseline['test_index']  # Already correct

# Join on data_row_index
merged = xgb.merge(baseline, on='data_row_index', suffixes=('_xgb', '_baseline'))
```

**Important Note on y_true Values**:

When `baseline.enabled: true` in a finetuning config (e.g., `experiments/configs/finetuning/finetuning.yaml`), baseline predictions are generated **alongside** XGBoost/TabPFN predictions using the **same winsorized y_true values**. The baseline model is evaluated within the finetuning experiment loop, after phase2 preprocessing, ensuring perfect alignment:

```yaml
# finetuning.yaml
baseline:
  enabled: true  # Baseline uses same winsorized targets as XGBoost/TabPFN
```

In this case, joining predictions will show **identical `y_true` values** across all models (baseline, xgboost, tabpfn).

**Legacy standalone baseline script**: The old `experiments/scripts/generate_baseline_results.py` generates baseline predictions WITHOUT phase2 preprocessing, resulting in:
- **Baseline `y_true`**: Raw log-transformed sale amounts (no winsorization)
- **Finetuning `y_true`**: Winsorized (clipped) at 1st/99th percentile of training data

This creates a mismatch (~3% of rows affected by winsorization). The standalone script is deprecated for finetuning comparisons—use the integrated baseline instead by setting `baseline.enabled: true` in your finetuning config.

### Mapping test_index to FIPS

To get the county (FIPS) for each prediction:

```python
import pyarrow.parquet as pq

# Load FIPS column from data.parquet
table = pq.read_table('data.parquet', columns=['fips'])
fips_all = table.column('fips').to_numpy()

# For baseline predictions (test_index is already data.parquet row index)
baseline['fips'] = fips_all[baseline['test_index'].values]

# For finetuning predictions (need to convert first)
xgb['data_row_index'] = unique_indices[xgb['test_index'].values]
xgb['fips'] = fips_all[xgb['data_row_index'].values]
```

**Why the difference?** The baseline script works directly with test_indices.npy and doesn't create a subset DataFrame. The finetuning code optimizes memory by loading only needed rows into a subset DataFrame, then uses iloc positions within that subset. Both approaches represent the same test samples, just with different indexing schemes.

---

## Contributing

When adding new experiments:
1. Use `CleanedDataLoader` to load preprocessed data
2. Apply Phase 2 preprocessing per train/test split (no leakage)
3. Check `log_transformed` flag from metadata for metrics
4. Update this README with usage examples

---

## License

[Add your license here]
