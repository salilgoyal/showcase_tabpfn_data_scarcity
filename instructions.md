# Pipeline Instructions

## Supported Models

Four models are available across all experiment types:

| Name in config | Description |
|---|---|
| `tabpfn` | TabPFN v2 (zero-shot, default) |
| `tabpfn_v2.5` | TabPFN v2.5 (gated HF repo — requires `HF_TOKEN` and license accepted at `huggingface.co/Prior-Labs/tabpfn_2_5`) |
| `tabicl` | TabICL v2 regressor (zero-shot, auto-downloads ~500MB checkpoint, no auth needed) |
| `xgboost` | XGBoost with Optuna hyperparameter search |

Add any combination to a config's `models:` list:
```yaml
models:
  - name: "tabpfn"
    enabled: true
  - name: "tabpfn_v2.5"
    enabled: true
  - name: "tabicl"
    enabled: true
  - name: "xgboost"
    enabled: false

tabpfn:
  version: "v2"   # do not change — version is fixed by the model name
  device: "cuda"

tabpfn_v2.5:
  device: "cuda"

tabicl:
  n_estimators: 8   # more = better but slower; 8 is default
  device: null      # null = auto-detect (cuda if available)

xgboost:
  optuna_trials: 50
  optuna_cv_folds: 3
  use_gpu: true
```

### Toggling models without editing YAML

Pass `--models` to override the config's `models:` section from the command line:

```bash
# Run only tabpfn_v2.5 and tabicl, ignoring the config's models list
python experiments/run_experiment.py --config <cfg.yaml> --models tabpfn_v2.5,tabicl

# Same via SLURM
sbatch experiments/slurm/nlp/geo_pooling.sh <cfg.yaml> --models tabpfn_v2.5,tabicl

# Same for all randsplits in a folder (avoids editing all 5 split configs)
bash experiments/scripts/submit_rand_splits_nlp.sh \
    experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits \
    --models tabpfn_v2.5,tabicl

# Or for ALL randsplit folders at once
bash experiments/scripts/submit_rand_splits_nlp.sh \
  experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits \
  --models tabpfn_v2.5,tabicl
```

---

Two experiments are documented here:

1. **Geo-pooling (random splits)** — compares geo-pooling (ratio 80%) vs no-pooling across 5 random train/test splits, visualized with `plot_line_by_train_size` in `notebooks/geo_pooling/play.ipynb`
2. **Finetuning sweep** — sweeps LoRA rank × learning rate × epoch size for globally finetuned TabPFN, with both pooling variants, visualized with `load_sweep_df()` + `plot_sweep_grid` in the same notebook

Both experiments ran on **Sherlock**. The NLP cluster has matching configs for Experiment 1 but its result directories are empty. All steps below assume Sherlock unless noted.

---

## Prerequisites

### Environment (Sherlock)

```bash
module load python/3.12 devel cmake/3.31.4 py-pyarrow/18.1.0_py312
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH
cd /home/users/salilg/tabpfn_data_scarcity
```

### Environment (NLP)

```bash
source /nlp/scr/salilg/miniconda3/bin/activate tabpfn_env
export PYTHONPATH=/sailhome/salilg/showcase_tabpfn_data_scarcity:$PYTHONPATH
cd /sailhome/salilg/showcase_tabpfn_data_scarcity
```

---

## Step 1: Phase 1 Preprocessing

**Skip if already done.** Check that `data.parquet` exists:
- Sherlock: `/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/data.parquet`
- NLP: `/nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/data.parquet`

If missing, run on Sherlock:

```bash
sbatch preprocessing/slurm/clean_data.sh preprocessing/configs/v2_no_onehot.yaml
```
Note that this command needs to be run on Sherlock as it uses data that is stored on Oak.

This reads raw county CSVs, cleans them, log-transforms the target, and saves a single pooled parquet file.

---

## Step 2: Generate Random Test/Train Splits

**Skip if already done.** Check that split directories `test_v4_rand_s{0..4}/` exist under the same preprocessed directory `/nlp/scr/salilg/showcase_property_tax/preprocessed/v2_no_onehot/`.

Run on Sherlock (interactive or via `generate_test_set_rand.sbatch`):

```bash
bash experiments/scripts/generate_test_set_rand.sh
```

This generates 5 independent random 80/20 splits of the same county set (selected by `experiments/configs/test_sets/test_v4_rand.yaml` with `county_seed=42`), one per seed. Output goes to:

```
/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s{0..4}/
```

For a SLURM-managed version:
```bash
sbatch experiments/slurm/splits/generate_test_set.sh \
    experiments/configs/test_sets/test_v4_rand.yaml \
    /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s0/ \
    /scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/data.parquet
```
(repeat for each seed, adjusting output dir and passing `--split_seed N` manually if needed)

---

## Experiment 0: Single-County Data Scaling

Trains TabPFN and XGBoost on varying amounts of data from a single county to produce learning curves. Cook County (FIPS 17031) is the default target.

- **Temporal split**: oldest 80% → train pool, most recent 20% → fixed test set
- **Sweep**: 22 train sizes (25–10,000) × 20 random seeds × 2 models = 880 trials
- **Phase 2 preprocessing**: winsorization (1st percentile), StandardScaler, median imputation — fit on each training sample independently
- **Ratio filter**: bottom 5% of assessed-to-sale ratio dropped per sale year

### Step 3: Submit job (NLP)

```bash
sbatch experiments/slurm/nlp/single_county_scaling.sh \
    experiments/configs/single_county_scaling/nlp/v2_no_onehot/cook_county_20seeds.yaml
```

Logs go to `logs/single_county_scaling/nlp/single_county_scaling/<job_id>.{out,err}`.

Results go to:
```
/nlp/scr/salilg/showcase_property_tax/results/single_county_scaling/v2_no_onehot/cook_county_20seeds/results.csv
```

Intermediate results are saved every 20 `(seed, train_size)` combos to `results_intermediate.csv` in the same directory.

### Step 4: Visualize

Open `notebooks/single_county_scaling/results.ipynb`. The notebook loads results directly from the path above and produces:
- 2×2 grid of learning curves (MAPE, MAE, RMSE, R²) with 95% CI bands
- Standalone MAPE learning curve
- Crossover table (at which train size does XGBoost catch up to TabPFN on mean MAPE)
- Per-seed scatter plot

### Adding a different county or more seeds

Copy and edit the config:

```yaml
single_county_scaling:
  county_fips: <FIPS>       # e.g. 6037 for LA County
  train_sizes: [25, 50, ...]
  seeds: [0, 100, ...]      # one seed = one random sample from the train pool
  test_fraction: 0.2
  temporal_split: true
  temporal_column: "sale_day"
  save_interval: 20

output:
  results_dir: "/nlp/scr/salilg/showcase_property_tax/results/single_county_scaling/v2_no_onehot/<your_name>/"
```

---

## Experiment 1: Geo-pooling (Random Splits)

### Step 3a: Submit Jobs (Sherlock)

```bash
bash experiments/scripts/submit_rand_splits_sherlock.sh
```

This submits 4-chunk SLURM array jobs for every YAML in `*_randsplits/` directories under `experiments/configs/geo_pooling/sherlock/v2_no_onehot/`. The two relevant families are:

- `test_v4_k40_ratio80_droplowest5_randsplits/` — 5 configs (s0–s4)
- `test_v4_k40_nopooling_droplowest5_randsplits/` — 5 configs (s0–s4)

Each config → 4 SLURM array tasks → each processes ~133 counties in parallel.

SLURM script used: `experiments/slurm/sherlock/geo_pooling.sh`

Results land in:
```
/scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/
  test_v4_rand_s{0..4}_geo_k40_ratio80_droplowest5/chunk_{0..3}/results.csv
  test_v4_rand_s{0..4}_geo_k40_nopooling_droplowest5/chunk_{0..3}/results.csv
```

### Step 3a (NLP alternative)

```bash
bash experiments/scripts/submit_rand_splits_nlp.sh
```

Uses `experiments/configs/geo_pooling/nlp/v2_no_onehot/` and `experiments/slurm/nlp/geo_pooling.sh`. Results go to `/nlp/scr/salilg/showcase_property_tax/results/geo_pooling/v2_no_onehot/`.

### Step 4a: Visualize

Open `notebooks/geo_pooling/play.ipynb` and run:

```python
exp_patterns = [
    'test_v4_rand_s*_geo_k40_ratio80_droplowest5',
    'test_v4_rand_s*_geo_k40_nopooling_droplowest5',
]
legible_exp_names = ['Pooling ratio 80%', 'No pooling']

df = create_concat_df_rand(exp_patterns, legible_exp_names)
plot_line_by_train_size(
    df, legible_exp_names,
    bucket_size=40,
    title_addition='SEs in each bucket computed over counties and 5 random train/test splits'
)
```

---

## Experiment 2: Finetuning Sweep

This runs entirely on Sherlock. It chains global finetuning jobs and geo-pooling jobs automatically via SLURM dependencies.

### Step 3b: Run the Sweep

From the Sherlock project root:

```bash
# Full sweep: 4 random seeds (96 FT jobs + 192 geo jobs)
python experiments/scripts/sweep.py --seeds 0 1 2 3

# Dry run to see all commands without submitting
python experiments/scripts/sweep.py --seeds 0 1 2 3 --dry-run

# If finetuning checkpoints already exist, skip to geo pooling only
python experiments/scripts/sweep.py --seeds 0 1 2 3 --geo-only
```

`sweep.py` does the following automatically:
1. Generates FT configs in `experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/`
2. Generates geo configs in `experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/`
3. Submits FT jobs via `experiments/slurm/sherlock/global_finetuning.sh`
4. Submits geo pooling jobs with `--dependency=afterok:<ft_job_id>` so they start automatically after finetuning completes, via `experiments/slurm/sherlock/geo_pooling.sh`

Hyperparameter grid swept: LRs `[1e-5, 5e-5, 1e-4, 5e-4]` × LoRA ranks `[4, 8, 16]` × epoch sizes `[50, 100]` × 4 seeds × 2 pooling variants (nopooling, ratio80).

FT results: `/scratch/users/salilg/property_tax/results/global_finetuning/v2_no_onehot/sweep/<name>/`
Geo results: `/scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/sweep/<name>_{nopooling,ratio80}/chunk_{0..3}/results.csv`

### Step 4b: Visualize

Open `notebooks/geo_pooling/play.ipynb` and run the cells under **"## Random seeds, sweep over finetuning choices"**:

```python
sweep_df = load_sweep_df()
plot_sweep_grid(sweep_df, bucket_size=40, df2=df2, ylim=(20, 150))
```

`df2` should be the concatenated rand-split DataFrame from Experiment 1 (used as a baseline comparison overlay).

# NOTE ON TABPFN ISSUE:
Job 15371711 loaded 264 results but only 132 completed keys — meaning only one model (xgboost) was in completed_keys, so tabpfn was re-runnable. Then 15371883 loaded the 396-result checkpoint (264 from 15370164 + 132 failures from 15371711) with still 132 completed keys, and re-ran tabpfn successfully. That explains the 3 entries.