# Files Used by Pipeline

## Shared by Both Experiments

### Library code
- `src/data/loading.py` — loads parquet data and groups by county
- `src/data/preprocessing.py` — Phase 2 preprocessing (winsorize, normalize, impute)
- `src/data/preprocessing_utils.py` — preprocessing helpers
- `src/data/geo_utils.py` — geographic distance / neighbor selection
- `src/data/filters.py` — ratio filter (drop bottom 5% by sale price ratio)
- `src/data/splitting.py` — train/test split logic
- `src/data/samplers.py` — neighbor budget sampling
- `src/data/column_categorizer.py` — column type classification
- `src/data/county_filter.py` — county-level filtering
- `src/models/tabpfn_wrapper.py` — TabPFN v2 wrapper
- `src/models/xgboost_wrapper.py` — XGBoost wrapper with Optuna HPO
- `src/models/base_model.py` — base model interface
- `src/evaluation/metrics.py` — R², MAE, RMSE, MAPE, MSE
- `src/evaluation/aggregation.py` — result aggregation
- `src/utils/paths.py` — path utilities

### Experiment runner
- `experiments/run_experiment.py` — main entry point, dispatches to experiment type
- `experiments/experiment_types/base.py` — base experiment class
- `experiments/experiment_types/__init__.py`
- `experiments/runners/base_runner.py` — chunked runner base class
- `experiments/runners/__init__.py`

### Data
- `data/us_county_latlng.csv` — county centroids (lat/lng) used for neighbor search

### Preprocessing (Phase 1)
- `preprocessing/scripts/clean_pooled_data.py` — reads raw county CSVs, cleans, log-transforms, saves parquet
- `preprocessing/configs/v2_no_onehot.yaml` — preprocessing config (label encoding, no one-hot)
- `preprocessing/slurm/clean_data.sh` — SLURM job script for preprocessing (Sherlock)
- `preprocessing/readme.md` — preprocessing documentation

### Test split generation
- `experiments/scripts/generate_test_set.py` — generates train/test indices for a county set
- `experiments/scripts/generate_test_set_rand.sh` — loops over seeds 0–4, calls generate_test_set.py
- `experiments/scripts/generate_test_set_rand.sbatch` — SLURM wrapper for the above
- `experiments/slurm/splits/generate_test_set.sh` — SLURM job script for a single split (Sherlock)
- `experiments/configs/test_sets/test_v4_rand.yaml` — county selection config (county_seed=42, random 80/20 split)

### Visualization notebook
- `notebooks/geo_pooling/play.ipynb` — all plots for both experiments

### Project
- `requirements.txt`
- `README.md`

---

## Experiment 1: Geo-pooling (Random Splits)

Compares `ratio80` (80% neighbor budget) vs `nopooling` across 5 random splits.

### Experiment type
- `experiments/experiment_types/geo_pooling.py` — core geo pooling logic (neighbor search, budget allocation, per-county training loop)

### Configs — Sherlock
- `experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_k40_ratio80_droplowest5_randsplits/` — 5 YAMLs (s0–s4)
- `experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits/` — 5 YAMLs (s0–s4)

### Configs — NLP (mirror, results empty)
- `experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_randsplits/` — 5 YAMLs (s0–s4)
- `experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_nopooling_droplowest5_randsplits/` — 5 YAMLs (s0–s4)

### Job submission
- `experiments/scripts/submit_rand_splits_sherlock.sh` — submits 4-chunk array jobs for all `*_randsplits/` configs (Sherlock)
- `experiments/scripts/submit_rand_splits_nlp.sh` — equivalent for NLP
- `experiments/slurm/sherlock/geo_pooling.sh` — SLURM array job script (Sherlock)
- `experiments/slurm/nlp/geo_pooling.sh` — SLURM array job script (NLP)

---

## Experiment 2: Finetuning Sweep

Sweeps LoRA rank × LR × epoch size for globally finetuned TabPFN, run on Sherlock only.

### Experiment types
- `experiments/experiment_types/global_finetuning.py` — trains a global TabPFN model on the full train pool via LoRA finetuning
- `experiments/experiment_types/geo_pooling.py` — same as Experiment 1; used to evaluate the finetuned checkpoints at inference time

### Additional model code
- `src/models/tabpfn_finetuning_v2.py` — TabPFN v2 LoRA finetuning implementation

### Sweep orchestration
- `experiments/scripts/sweep.py` — generates all configs and submits FT + geo jobs with SLURM dependencies

### Base config read by sweep.py
- `experiments/configs/global_finetuning/sherlock/v2_no_onehot/internal_15k_percounty.yaml` — template FT config

### Generated configs (produced by sweep.py, committed to repo)
- `experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/` — 120 FT configs (lora{4,8,16} × lr{1e-5,5e-5,1e-4,5e-4} × ep{50,100} × {temporal, s0–s3})
- `experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/` — 240 geo configs (above × {nopooling, ratio80})

### SLURM job scripts
- `experiments/slurm/sherlock/global_finetuning.sh` — runs a single global finetuning job
- `experiments/slurm/sherlock/geo_pooling.sh` — runs a single geo pooling array job (shared with Experiment 1)
