# TabPFN Data Scarcity Experiments

Compare TabPFN and XGBoost on property sale price prediction across US counties, with a focus on **data scarcity**: many counties have very few historical transactions. The central question is whether using data from geographically nearby counties, and/or using tabular foundation models, improves predictions for small counties.

**Models**:
- **TabPFN** — transformer doing in-context learning (zero-shot or globally finetuned)
- **XGBoost** — Optuna-tuned gradient boosting baseline

**Target**: `SALE_AMOUNT` (log-transformed property sale price)

---

## End-to-End Pipeline

```
Raw county CSVs (~2667 counties)
        │
        ▼ Phase 1 preprocessing (run once)
        │   preprocessing/scripts/clean_pooled_data.py
        │
data.parquet  (~15 GB, all counties pooled and cleaned)
        │
        ▼ Test set generation (run once per version)
        │   experiments/scripts/generate_test_set.py
        │
test_v4/
  ├── test_indices.npy          # which rows are held out for evaluation
  └── train_pool_indices.npy    # which rows are available for training
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  geo_pooling                            per_county_scaling
  (builds per-county training sets       (builds per-county training sets
   dynamically from own + neighbor        dynamically from own data only)
   train_pool data)
        ▲
        │  (optional: load finetuned checkpoint instead of zero-shot TabPFN)
        │
  global_finetuning
  (samples from train_pool, finetunes
   TabPFN, saves model.pt checkpoint)
        │
        ▼
Results on scratch (results.csv, predictions.parquet, neighbor_usage.parquet)
```

**Phase 2 preprocessing** (winsorization, StandardScaler, median imputation) does not happen in a separate step. It runs inside each experiment, fit on the training data only and then applied to both train and test. This is what prevents data leakage.

---

## Step 1: Phase 1 Preprocessing (run once)

**Script**: `preprocessing/scripts/clean_pooled_data.py`
**Config**: `preprocessing/configs/v2_no_onehot.yaml` (current active version)
**SLURM**: `preprocessing/slurm/clean_data.sh` (128 GB RAM, ~4 hours)

```bash
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v2_no_onehot.yaml
```

Phase 1 applies all transformations that don't depend on a train/test split, so they're safe to do globally:

- Load all county CSVs
- Drop null labels, single-value columns, mostly-null columns (< 50% coverage)
- Drop suspicious transactions (lowest 5% of assessed-value-to-sale-price ratio and top 1% [double check])
- Drop repeat sales (keep only the most recent sale per parcel)
- Generate temporal features from `sale_date` (year, month, day, day-of-week, days since 2000)
- Label-encode categorical columns (no one-hot encoding)
- Log-transform the target (`SALE_AMOUNT`)

**Output** at `/nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/`:
```
v2_no_onehot/
├── data.parquet          # ~15 GB, all counties, cleaned
├── metadata.json         # column types, log_transformed flag, row counts
├── config.yaml           # copy of config used
└── preprocessing_log.txt # detailed log
```

---

## Step 2: Test Set Generation (run once per version)

**Script**: `experiments/scripts/generate_test_set.py`
**Config**: `experiments/configs/test_sets/` (e.g. `test_v4.yaml`, `test_v4_rand.yaml`)

```bash
python experiments/scripts/generate_test_set.py \
    --config experiments/configs/test_sets/test_v4.yaml \
    --data_path /nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/data.parquet \
    --output_dir /nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/test_v4/
```

This selects which counties are in the test set and how each county's data is split between training and test. The split is recorded as **row indices into data.parquet**, so it's reproducible without re-running.

**County selection**: size-stratified sampling across five buckets (tiny: 2–100 rows, small: 100–500, medium: 500–2K, large: 2K–10K, xlarge: 10K+). 

**Within-county split** (for `test_v4`): temporal 80/20 — the oldest 80% of each county's transactions go into the training pool, the most recent 20% are held out as test. The training pool is NOT a fixed set of rows you train on — it's the universe of rows any experiment is allowed to sample from. Each experiment builds its own training set from this pool at runtime.

**Output** at e.g. `/nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/test_v4/`:
```
test_v4/
├── test_indices.npy       # data.parquet row indices for test samples
├── train_pool_indices.npy # data.parquet row indices available for training
├── test_counties.json     # list of test county FIPS codes
├── county_info.json       # per-county statistics (total rows, test rows, size bucket)
├── size_buckets.json      # which counties are in each bucket
├── metadata.json          # split config and summary statistics
└── summary_report.txt     # human-readable summary
```

### Active test set versions

| Version | Counties | Test samples | Split |
|---------|----------|-------------|-------|
| `test_v4` | 525 | ~25K | Temporal 80/20 (oldest 80% → train pool, newest 20% → test) |
| `test_v4_rand_s0` … `test_v4_rand_s4` | 525 | ~25K | Random 80/20, seeds 0–4 |

The random splits exist to evaluate variance across different train/test configurations. Temporal (`test_v4`) is the primary evaluation.

---

## Step 3: Experiments

All experiments share this pattern at runtime:

1. Load `test_indices.npy` and `train_pool_indices.npy` (from the test set directory)
2. Load only the needed rows from `data.parquet` (memory-efficient)
3. Build training data dynamically (each experiment does this differently — see below)
4. Apply Phase 2 preprocessing: fit winsorizer + StandardScaler + median imputer on training data, transform both train and test
5. Train models, evaluate on held-out test rows, save results

**Phase 2 steps** (configured in experiment YAML under `preprocessing.phase2_steps`):
- Winsorization: clip outlier values at train 1st/99th percentile
- StandardScaler normalization: fit on train mean/std
- Median imputation: fill NaNs using train medians

Additionally, a **ratio filter** (configured under `ratio_filter`) can drop suspicious rows where `MARKET_TOTAL_VALUE / exp(SALE_AMOUNT)` is in the bottom N% (computed within each sale year). This runs before Phase 2.
[double chekc if this was done in phase 2 or phase 1 preprocessing]

---

### Experiment: Geographic Pooling (`geo_pooling`)

**Goal**: Train a per-county model using the county's own historical data plus data from geographically nearby counties. Test whether geographic pooling improves predictions for small counties.

**Implementation**: `experiments/experiment_types/geo_pooling.py`
**SLURM**: `experiments/slurm/nlp/geo_pooling.sh` / `experiments/slurm/sherlock/geo_pooling.sh`
**Configs**: `experiments/configs/geo_pooling/nlp/v2_no_onehot/` (NLP) or `sherlock/v2_no_onehot/` (Sherlock)

**How it works, per county**:

1. Take the county's own train pool data (all of it — no subsampling of own data)
2. Find the k nearest neighbor counties by geographic centroid distance (using `data/us_county_latlng.csv`)
3. Fill a "neighbor budget" (= `own_train_size × neighbor_budget_ratio`, capped at `max_total_training_size - own_train_size`) by sampling from neighbors closest-first. Typically set to be 80% of the target county's train split size. [NOTE: it'd probably be better if we did this for a few different budgets and showed that it doesn't matter etc.]
4. Combine own data + neighbor data → apply Phase 2 → train models
5. Evaluate on the county's held-out test set

The run is parallelized as a SLURM array job. Each array task processes a contiguous chunk of counties and writes results to its own subdirectory (`chunk_0/`, `chunk_1/`, ...).

**Running**:
```bash
# Array job: split counties into 4 chunks
N_CHUNKS=4 sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
    experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5.yaml
```

**Key config parameters** (`geo_pooling` section):
```yaml
geo_pooling:
  centroids_csv: "data/us_county_latlng.csv"
  max_k_neighbors: 40            # Maximum number of neighbor counties to consider
  neighbor_budget_ratio: 0.8     # Neighbor data = 0.8 × own train size
  max_total_training_size: 10000 # Hard cap on own + neighbor combined (TabPFN limit)
```

**Output**:
```
results/geo_pooling/v2_no_onehot/test_v4_geo_k40_ratio80_droplowest5/
├── chunk_0/
│   ├── results.csv              # Per-county metrics: fips, model, r2, mae, rmse, mape, ...
│   ├── results_checkpoint.csv   # Same, written periodically for fault tolerance
│   ├── completed_keys.pkl       # Set of (fips, model) pairs already done
│   └── neighbor_usage.parquet   # Which neighbors contributed how many rows, for each county
├── chunk_1/ ...
└── experiment.log
```

**Checkpointing**: By default, jobs resume automatically (checkpoint loaded on startup). To start fresh, set `checkpointing.resume: false` in the config.

#### Variant: Extended Neighbor Pool

By default, neighbors are restricted to the 525 test_v4 counties. Set `extend_neighbor_pool_beyond_testv4: true` to draw neighbors from all ~2667 counties in the dataset. The extra rows are loaded from `data.parquet` at runtime.

```yaml
geo_pooling:
  extend_neighbor_pool_beyond_testv4: true
```

#### Variant: Diversified Neighbor Sampling

When the nearest neighbor county alone has more data than the entire neighbor budget, the default greedy approach takes all data from that one county. Enable diversification to instead split the budget equally among the N nearest neighbors:

```yaml
geo_pooling:
  diversify_neighbors: true
  diversify_n: 3
```

#### Variant: Using a Globally Finetuned TabPFN

Instead of zero-shot TabPFN (pure in-context learning), use a checkpoint produced by the `global_finetuning` experiment:

```yaml
models:
  - name: "tabpfn_global_finetuned"
    enabled: true
    checkpoint_dir: "/nlp/scr/salilg/showcase_property_tax/results/global_finetuning/v2_no_onehot/internal_15k/"
  - name: "xgboost"
    enabled: true
```

The finetuned model still uses in-context learning at inference — it just starts from a TabPFN checkpoint that has been adapted to property data.

---

### Experiment: Global Finetuning (`global_finetuning`)

**Goal**: Finetune TabPFN once on a large pool of property data, save the checkpoint, and use it in geo_pooling experiments as a better-initialized base model.

**Implementation**: `experiments/experiment_types/global_finetuning.py`
**Model**: `src/models/tabpfn_finetuning_v2.py`
**SLURM**: `experiments/slurm/nlp/global_finetuning.sh` / `experiments/slurm/sherlock/global_finetuning.sh`
**Configs**: `experiments/configs/global_finetuning/nlp/v2_no_onehot/` (NLP) or `sherlock/v2_no_onehot/`

**Two data variants**:
- **Internal** (`variant: "internal"`): Sample from the train_pool of test_v4 counties (the historical data from the same counties we'll evaluate on). Uses only the training-split rows, so there's no test leakage.
- **External** (`variant: "external"`): Sample from counties that are NOT in test_v4. Completely disjoint from the evaluation set.

**How it works**:

1. Load test set (to know which counties are in test_v4)
2. Depending on variant: sample ~15K rows from train_pool (internal) or from external counties (external)
3. Apply ratio filter and Phase 2 preprocessing (winsorization + imputation; StandardScaler is skipped for `per_county` training mode since normalization is applied per-county inside the training loop)
4. Finetune TabPFN using a Yandex-style direct training loop on the raw checkpoint weights
5. Save the finetuned checkpoint to disk

**Running**:
```bash
sbatch experiments/slurm/nlp/global_finetuning.sh \
    experiments/configs/global_finetuning/nlp/v2_no_onehot/internal_15k.yaml
```

**Key config parameters**:
```yaml
global_finetuning:
  variant: "internal"     # "internal" (test_v4 train pool) or "external" (non-test_v4 counties)
  n_samples: 15000        # Rows to sample for finetuning
  sampling_strategy: "uniform"  # "uniform" or "stratified_by_county"

finetuning:
  learning_rate: 1e-4
  max_epochs: 100
  patience: 16            # Early stopping
  epoch_size: 10          # Steps per epoch
  seq_len_pred: 1024      # Query samples per step
  max_context_size: 10000 # Cap on context size (to avoid OOM)
  val_fraction: 0.2       # Internal validation split
  finetune_mode: "full"   # "full" or "lora" (use lora_rank to set LoRA rank)
```

**Output**:
```
results/global_finetuning/v2_no_onehot/internal_15k/
├── model.pt              # Finetuned model weights (load with DirectFineTunedTabPFNModel.load_from_disk())
├── metadata.json         # Column info, target stats, config used
├── transforms.pkl        # Target/prediction transforms
├── training_history.json # Per-epoch train loss and val R² (for plotting)
└── results.csv           # Summary row: best_epoch, best_val_loss, total time, etc.
```

**Finetuning implementation note**: The v2 implementation loads the TabPFN checkpoint directly (bypassing the high-level `TabPFNRegressor` API) so optimizer parameter references stay stable across training steps. An earlier v1 implementation (`tabpfn_finetuning.py`, now deleted) had a bug where parameters were never updated because `fit_from_preprocessed()` recreated internal objects on each call, breaking optimizer references.

**Typical workflow** (finetune then run geo_pooling):
```bash
# Step 1: Finetune (saves checkpoint)
sbatch experiments/slurm/nlp/global_finetuning.sh \
    experiments/configs/global_finetuning/nlp/v2_no_onehot/internal_15k.yaml

# Step 2: Geo pooling using the finetuned checkpoint
N_CHUNKS=4 sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
    experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_internal.yaml
```

**Plotting training curves**:
```python
import json, matplotlib.pyplot as plt

with open('.../training_history.json') as f:
    history = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_losses']); ax1.set(xlabel='Epoch', ylabel='Train loss')
ax2.plot([m.get('r2') for m in history['val_metrics']]); ax2.set(xlabel='Epoch', ylabel='Val R²')
plt.tight_layout(); plt.show()
```

---

### Experiment: Per-County Scaling (`per_county_scaling`)

**Goal**: Build learning curves for small counties using only their own historical data. Understand how much training data a county needs, and where TabPFN vs XGBoost cross over.

**Implementation**: `experiments/experiment_types/per_county_scaling.py`
**Configs**: `experiments/configs/per_county_scaling/nlp/` or `sherlock/`

**How it works**:

For each county × train_size × seed × model combination:

1. Sample `train_size` rows from the county's own train pool
2. Apply Phase 2 preprocessing (fit fresh on this sample)
3. Train the model and evaluate on the county's held-out test set
4. Record metrics

Checkpoints every 50 combos; resumes automatically if interrupted.

**Running** (SLURM array — one task per county):
```bash
# Each array task = one county FIPS
sbatch --array=... experiments/slurm/nlp/per_county_scaling.sh \
    experiments/configs/per_county_scaling/nlp/test_v4.yaml
```

Or test a single county locally:
```bash
python experiments/run_experiment.py \
    --experiment_type per_county_scaling \
    --config experiments/configs/per_county_scaling/nlp/test_v4.yaml \
    --county_fips 31007
```

**Key config parameters**:
```yaml
target_buckets: ["tiny", "small", "medium"]
train_sizes: [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500]
n_seeds: 1
```

**Output** (one subdirectory per county):
```
results/per_county_scaling/test_v4/
├── county_31007/
│   ├── results.csv            # All train_size × seed × model rows
│   ├── results_checkpoint.csv # Checkpoint copy
│   └── completed_keys.pkl     # For resume
├── county_31009/
│   └── ...
└── experiment.log
```

**Result columns**: `fips`, `size_bucket`, `county_train_pool_size`, `county_test_size`, `requested_train_size`, `actual_train_size`, `seed`, `model`, `r2`, `mae`, `rmse`, `mape`

**Quick smoke test** (~2–3 minutes, no GPU needed):
```bash
python experiments/scripts/smoke_test_per_county_scaling.py
```

---

## Repository Structure

```
showcase_tabpfn_data_scarcity/
│
├── preprocessing/                   # Phase 1: clean raw data (run once)
│   ├── scripts/
│   │   └── clean_pooled_data.py     # Main Phase 1 script
│   ├── configs/
│   │   └── v2_no_onehot.yaml        # Current preprocessing config
│   ├── slurm/
│   │   └── clean_data.sh            # SLURM job (128 GB, 4 hours)
│   └── analysis/                    # Data quality analysis outputs
│
├── experiments/                     # All experiments
│   ├── run_experiment.py            # Main CLI entry point
│   ├── experiment_types/
│   │   ├── base.py                  # Abstract base class
│   │   ├── geo_pooling.py           # Geographic pooling experiment
│   │   ├── global_finetuning.py     # Global TabPFN finetuning
│   │   └── per_county_scaling.py    # Per-county learning curves
│   ├── configs/
│   │   ├── geo_pooling/             # ~378 YAML configs (nlp/ and sherlock/)
│   │   ├── global_finetuning/       # ~138 YAML configs (nlp/ and sherlock/)
│   │   ├── per_county_scaling/      # 3 YAML configs
│   │   └── test_sets/               # Test set configs (test_v4.yaml, test_v4_rand.yaml)
│   ├── scripts/
│   │   ├── generate_test_set.py     # Generate test/train-pool split
│   │   ├── sweep.py                 # Hyperparameter sweep (LoRA × LR × epochs)
│   │   ├── smoke_test_per_county_scaling.py
│   │   └── recompute_per_county_mape.py
│   └── slurm/
│       ├── nlp/
│       │   ├── geo_pooling.sh
│       │   └── global_finetuning.sh
│       └── sherlock/
│           ├── geo_pooling.sh
│           └── global_finetuning.sh
│
├── src/                             # Reusable library
│   ├── data/
│   │   ├── loading.py               # CleanedDataLoader (reads data.parquet)
│   │   ├── preprocessing_utils.py   # Phase 2 (winsorize, scale, impute)
│   │   ├── split_strategies.py      # Test set generation logic
│   │   ├── geo_utils.py             # k-nearest-neighbor by lat/lng
│   │   └── filters.py               # County and feature filtering
│   ├── models/
│   │   ├── tabpfn_wrapper.py        # Zero-shot TabPFN
│   │   ├── tabpfn_finetuning_v2.py  # Finetuned TabPFN (current)
│   │   ├── xgboost_wrapper.py       # XGBoost with Optuna tuning
│   │   └── tabpfn_lib/              # Low-level TabPFN architecture components
│   └── evaluation/
│       └── metrics.py               # R², MAE, RMSE, MAPE
│
├── data/                            # Static metadata (committed to repo)
│   ├── us_county_latlng.csv         # County centroids for geo distance
│   ├── county_row_counts.csv        # Row counts per county
│   ├── county_metadata/             # County lists (high-coverage set)
│   └── feature_lists/               # Feature lists by county set
│
├── notebooks/
│   └── geo_pooling/                 # Analysis notebooks
│
└── logs/                            # SLURM job logs (gitignored)
    ├── geo_pooling/
    ├── global_finetuning/
    └── per_county_scaling/
```

---

## Models

### TabPFN (zero-shot)

In-context learning: the training set is passed as context at inference time, no gradient updates. Fast and hyperparameter-free, but capped at 10K context samples and ~100 features.

**Implementation**: `src/models/tabpfn_wrapper.py`

### TabPFN (globally finetuned)

Same architecture, but weights have been adapted to property data via a supervised finetuning loop. Loaded from a checkpoint produced by `global_finetuning`. Still uses in-context learning at inference — only the base weights differ.

**Implementation**: `src/models/tabpfn_finetuning_v2.py`

The finetuning loop (Yandex-style):
```
For each epoch (epoch_size steps):
  Sample seq_len_pred query indices
  Context = random sample of min(max_context_size, N_train) rows
  Forward: features → targets → transformer → bar-distribution decoder (5000 bins)
  Loss: bar distribution NLL on query samples
  Backward → gradient clip → optimizer step
Validation: full train as context, predict val set, compute R²
Early stopping: restore best checkpoint after `patience` epochs without improvement
```

### XGBoost

Hyperparameter tuning via Optuna (50 trials, 3-fold CV by default). CV folds are automatically reduced to 2 when `train_size < 30`.

**Implementation**: `src/models/xgboost_wrapper.py`

---

## Data Storage

| Content | NLP cluster path | Sherlock path |
|---------|-----------------|---------------|
| Raw county CSVs | `/nlp/scr/salilg/showcase_property_tax/county_csvs/` | `/scratch/users/salilg/property_tax/county_csvs/` |
| Preprocessed data | `/nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/` | `/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/` |
| Test set (test_v4) | `.../preprocessed/v2_no_onehot/test_v4/` | same pattern |
| Results | `/nlp/scr/salilg/showcase_property_tax/results/` | `/scratch/users/salilg/property_tax/results/` |

Results are organized by experiment type and config name:
```
results/
├── geo_pooling/v2_no_onehot/<config_name>/chunk_0/results.csv
├── global_finetuning/v2_no_onehot/<config_name>/model.pt
└── per_county_scaling/<config_name>/county_<fips>/results.csv
```

---

## SLURM Quick Reference

| Job type | Memory | Time | GPU | Script |
|----------|--------|------|-----|--------|
| Phase 1 preprocessing | 128 GB | 4 h | No | `preprocessing/slurm/clean_data.sh` |
| Test set generation | — | — | No | run locally |
| Global finetuning | 64 GB | 6 h | 1× | `experiments/slurm/nlp/global_finetuning.sh` |
| Geo pooling (per chunk) | 64 GB | 12 h | 1× | `experiments/slurm/nlp/geo_pooling.sh` |

```bash
# Monitor jobs
squeue -u $USER

# View logs
tail -f logs/geo_pooling/nlp/<jobname>/<jobid>_<taskid>.out
tail -f logs/global_finetuning/nlp/<jobname>_<jobid>.out

# Cancel
scancel <job_id>
```

---

## Hyperparameter Sweep

The sweep script runs a grid search over finetuning hyperparameters, with geo_pooling jobs queued as SLURM dependencies:

```bash
# Dry run: print all commands without submitting
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --dry-run

# Full sweep: temporal + 4 random seeds (120 FT jobs + 240 geo pooling jobs)
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3

# Finetuning only
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --ft-only

# Geo pooling only (when checkpoints already exist)
python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --geo-only
```

Grid: `learning_rate` ∈ {1e-5, 5e-5, 1e-4, 5e-4} × `lora_rank` ∈ {4, 8, 16} × `epoch_size` ∈ {50, 100}.

Generated configs go under `experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/` and `experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/`.

---

## Troubleshooting

**Phase 2 double-scaling**: `global_finetuning` with `training_mode: "per_county"` skips StandardScaler during Phase 2 (sets `normalize_continuous: false` internally). This is intentional — the training loop applies per-county normalization on each step, matching what geo_pooling does at inference. Without this, data would be scaled twice.

**Loss spikes during finetuning**: Can occur when a context split draws 2 samples with nearly identical log prices, producing `y_ctx_std ≈ 1e-8` and blowing up normalized targets. The fix (clamping `y_norm` to `[-10, 10]`) is in `src/models/tabpfn_finetuning_v2.py`. Use `spike_diagnostics: true` in the finetuning config to log per-step diagnostics.

**OOM during finetuning**: Set `max_context_size` in the finetuning config to cap how many rows are used as context per step (e.g. 10000 on an 80 GB A100).

**Geo pooling resume**: Add `checkpointing.resume: true` to the config to resume a partially completed run. By default this is already `true`; omitting it or setting it to `false` restarts from scratch.
