# Finetuning Debugging

## Background

Two finetuning strategies were tested as alternatives to zero-shot geo-pooled TabPFN:

1. **Per-county finetuning**: fine-tune a separate TabPFN model for each county using its own train pool.
2. **Global finetuning** (Yandex-style ICL finetuning): fine-tune a single TabPFN once on ~14K samples drawn from the temporal train pool across all counties, then use it as a drop-in replacement in the geo-pooling pipeline.

**Results from the geo-pooling experiments** (random splits, k=40 neighbors, ratio80, internal checkpoint):
- Per-county finetuning: ~same MAPE as XGBoost, worse than zero-shot TabPFN
- Global finetuning: catastrophic on small/rural counties (MAPE ~280 vs ~45 for zero-shot)

The diagnostics in this folder investigate why.

---

## Root Cause (confirmed)

**ICL collapse correlates with n_cont** (number of non-NaN float columns in a county's own train pool).

Counties whose own train pool has many all-NaN columns (small/rural counties where many features were never recorded) produce a heavily zero-padded context tensor — the finetuned model reads mostly zeros and collapses toward its global prior (log price ≈ 11.59). Counties with nearly full feature coverage (n_cont ≈ 55–57) show intact ICL.

Quantitatively (job 14821800):

| FIPS | Bucket | n_cont | FT Normal MAPE | FT Shuffled MAPE | ICL intact? |
|------|--------|--------|----------------|------------------|-------------|
| 48229 | tiny   | 31 | 262.9 | 270.1 | No — nearly identical |
| 27107 | medium | 47 | 238.1 | **238.1** | No — completely identical |
| 28115 | tiny   | 50 | 100.5 | 108.4 | Barely |
| 48045 | tiny   | 19 | 48.1  | 54.4  | Partially |
| 28101 | medium | 55 | **24.8** | 68.0 | Yes — comparable to zero-shot |
| 13099 | medium | 56 | 63.4  | 212.0 | Yes — strongly reads context |

The reason n_cont differs per county: Phase 2 preprocessing computes per-county statistics (median) to impute missing values. For tiny counties with only 25 own-train samples, many feature columns are entirely NaN — `median()` = NaN, `fillna(NaN)` = no-op — so those columns stay NaN in the tensor. This is harmless in the full geo-pooling pipeline because the training context includes k=40 neighbor counties, giving enough data for every column to have a valid median. The diagnostic uses only the county's own train pool, exposing the issue.

**Key insight**: The global finetuned model performs fine for counties with dense feature coverage. The catastrophic behavior on small counties is primarily a feature-sparsity problem, not a catastrophic ICL forgetting problem.

---

## Bug Fixes Made

### 1. `src/models/tabpfn_finetuning_v2.py` — NaN propagation fix
In `_prepare_features()`, continuous features were converted to tensors without NaN handling. All-NaN columns (from small-county feature sparsity) propagated NaN through the transformer, producing all-NaN predictions.

**Fix**: added `.fillna(0.0)` before tensor conversion. After Phase 2 StandardScaler normalization, 0 ≈ column mean, so this is a safe neutral imputation. Categoricals already had `np.where(np.isnan(...), 0, ...)` — continuous now matches.

```python
# Before:
x_num = torch.tensor(X[present_continuous].values, dtype=torch.float32)
# After:
x_num = torch.tensor(X[present_continuous].fillna(0.0).values, dtype=torch.float32)
```

---

## Files in this folder

| File | Purpose |
|------|---------|
| `diag_context_sensitivity.py` | Tier 1: tests Normal / Shuffled-y / Minimal context on 3 tiny + 3 medium counties. Confirms whether the model reads its ICL context at all. |
| `diag_distribution_swap.py` | Tier 2: swaps the county's own context for a large sample from the finetuning distribution. Tests whether ICL works when context is in-distribution. |
| `diag_pipeline_comparison.py` | Tier 3: compares 3 pipelines on ~30 counties to isolate whether the MAPE gap is from finetuned weights or from the different preprocessing pipeline. See interpretation guide in the script docstring. |
| `run_diagnostic.sh` | Generic NLP SLURM script. Pass any Python script + args as positional arguments. |
| `instructions.md` | How to run the scripts, interpret output, and use the decision tree. |
| `analysis.ipynb` | Notebook for visualizing results. Sections 1-4: n_cont analysis, diagnostic CSV, geo-pooling experiments. **Section 5**: per-sample prediction distribution comparison (scatter plots, residual histograms, bias/spread analysis). Needs ~2 GB RAM; run on a SLURM node or interactive job. |

### Output files (in `logs/debugging/finetuning/`)

| File | Contents |
|------|---------|
| `ft_diagnostic_<jobid>.out` | Full console output from a diagnostic run |
| `ft_diagnostic_<jobid>.err` | Numpy RuntimeWarnings (harmless — from all-NaN column median computation in Phase 2) |
| `diag_context_sensitivity_<jobid>.csv` | Per-FIPS results: fips, bucket, own_train_size, test_size, n_cont, model, context, mape, r2, mae, rmse |
| `diag_distribution_swap_<jobid>.csv` | Per-FIPS results: fips, bucket, own_train_size, test_size, model, context_source, n_context_samples, mape, r2, mae |
| `diag_pipeline_comparison_<jobid>.csv` | Per-FIPS results: fips, n_cont, own_train_size, test_size, model (zs_regressor/zs_ft_pipeline/ft_pipeline), mape, r2, mae, rmse |

---

## Running the diagnostics

```bash
# Tier 1 (run first)
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_context_sensitivity.py

# Tier 2
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_distribution_swap.py

# Tier 3: Pipeline comparison (isolates weights vs preprocessing)
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_pipeline_comparison.py
```

See `instructions.md` for the full decision tree and interpretation guide.

---

## Findings (updated)

### n_cont is NOT the main driver

The 6-county Tier 1 diagnostic suggested n_cont (feature sparsity) was the root cause. However, analysis at scale (Section 3-5 of `analysis.ipynb`) across all 525 test counties shows:

- **Finetuned model is uniformly ~15% worse** across ALL n_cont bins (ratio 1.15–1.18)
- Zero-shot TabPFN and XGBoost show flat MAPE regardless of n_cont
- The degradation is **systematic**, not limited to low-n_cont counties

### Key differences between pipelines

The zero-shot and finetuned models use different preprocessing pipelines at inference time:
- **Zero-shot**: `TabPFNRegressor` with its own internal feature detection and y-standardization (per-county)
- **Finetuned**: `DirectFineTunedTabPFNModel._prepare_features()` with global y_mean/y_std from the finetuning dataset

The **Tier 3 diagnostic** (`diag_pipeline_comparison.py`) isolates whether the gap comes from the weight change or the pipeline change by running original (unfinetuned) weights through the finetuning pipeline.

---

## Next steps (open questions)

1. **Run Tier 3** (`diag_pipeline_comparison.py`) to isolate weights vs pipeline.
2. **Run analysis.ipynb Section 5** to examine per-sample prediction distributions (bias, spread, failure modes).
3. **Potential fixes** (depending on Tier 3 results):
   - If pipeline is the issue: align feature processing and y-standardization between pipelines
   - If weights are the issue:
     - LoRA (preserves base model ICL capability, only adapts a small number of parameters)
     - Regularization / LR warmup to reduce catastrophic forgetting
     - Epoch sweep (test earlier checkpoints — the optimal epoch for global validation may overfit for per-county ICL)
     - County-level y-normalization at inference (shift y_context by county mean before passing to model)
