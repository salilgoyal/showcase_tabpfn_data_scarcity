# Finetuning Diagnostics

Two scripts to diagnose why the globally finetuned TabPFN model performs catastrophically on small counties (MAPE ~280 vs ~45 for zero-shot).

**Hypothesis**: Finetuning broke the model's ICL mechanism — it ignores the context provided at inference time and predicts near the global prior (mean log price from the 14K finetuning samples ≈ 11.59).

---

## Running the diagnostics

### Tier 1: Context sensitivity (run this first)

```bash
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_context_sensitivity.py

# To test the external checkpoint instead:
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_context_sensitivity.py \
  --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/external_15k/
```

Tests 3 context variants on 3 tiny + 3 medium counties, for both zero-shot and globally finetuned TabPFN:
- **Normal**: county's own train pool as context
- **Shuffled-y**: same X context, but y labels randomly permuted (context becomes uninformative)
- **Minimal**: only 2 random samples as context

### Tier 2: Distribution swap (run after Tier 1)

```bash
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_distribution_swap.py

# With more swap samples or external checkpoint:
sbatch debugging/finetuning/run_diagnostic.sh \
  debugging/finetuning/diag_distribution_swap.py \
  --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/external_15k/ \
  --n_swap_samples 3000
```

For each county, compares 4 conditions:
- Zero-shot + county's own context
- Finetuned + county's own context
- Finetuned + finetuning-distribution context (random sample from all train pool data)
- Zero-shot + finetuning-distribution context

Logs: `logs/debugging/finetuning/ft_diagnostic_<jobid>.out`

---

## Analyzing the output

### Tier 1 output

Each county prints a table like:

```
County FIPS=XXXXX (tiny, own_train_size=25, test_size=8)
  y_train mean (log): 10.20,  global prior (y_mean): 11.59

  Model                          | Context      | MAPE  | R²
  TabPFN (zero-shot)             | Normal       | 42.3  | 0.71
  TabPFN (zero-shot)             | Shuffled-y   | 89.1  | -0.24
  TabPFN (zero-shot)             | Minimal (2)  | 78.5  | -0.11
  TabPFN (globally finetuned)    | Normal       | 285.0 | -2.15
  TabPFN (globally finetuned)    | Shuffled-y   | 290.2 | -2.20
  TabPFN (globally finetuned)    | Minimal (2)  | 282.0 | -2.10

  Predictions (globally finetuned, normal context, first 10):
    y_pred (log): [11.58, 11.61, 11.55, ...]
    y_true (log): [10.20, 10.45, 9.88, ...]
```

**What to look for:**

| Observation | Interpretation | Next step |
|---|---|---|
| Finetuned MAPE is similar across Normal / Shuffled-y / Minimal | Model ignores context entirely — ICL broken | Tier 2 to confirm severity; rethink finetuning approach |
| Finetuned MAPE degrades with Shuffled-y (like zero-shot does) | Model reads context, failure is elsewhere | Inspect predictions for distribution mismatch |
| Predictions cluster near global prior (≈ 11.59) for cheap small counties | Prior-collapse: model stopped adapting to context | Confirms ICL failure; likely needs LoRA or regularization |
| Finetuned works well on medium counties but fails on tiny ones | Distribution-specific ICL: works only for in-distribution inputs | Tier 2 to test whether training-distribution context helps |
| Zero-shot also degrades with Shuffled-y (expected) | Sanity check passes — zero-shot still reads context normally | |

**Key number**: the gap between finetuned Normal MAPE and finetuned Shuffled-y MAPE. If gap < 10%, the model is not reading the context.

---

### Tier 2 output

Each county prints a table like:

```
County FIPS=XXXXX (tiny, own_train_size=25)
  y_train mean (log): 10.20, y_swap mean (log): 11.62, global prior: 11.59

  Model                          | Context source          | MAPE  | R²
  TabPFN (zero-shot)             | County own (25)         | 42.3  | 0.71
  TabPFN (globally finetuned)    | County own (25)         | 285.0 | -2.15
  TabPFN (globally finetuned)    | Finetuning dist (2000)  | 48.2  | 0.68   <-- or still bad?
  TabPFN (zero-shot)             | Finetuning dist (2000)  | 95.0  | 0.21
```

**What to look for:**

| Observation | Interpretation | Next step |
|---|---|---|
| Finetuned + finetuning dist dramatically better (MAPE drops from ~285 to ~50) | ICL works, but only for in-distribution context | Try LoRA, or finetune on more diverse data (all 2667 counties) |
| Finetuned + finetuning dist still bad (MAPE ~280) | ICL mechanism broadly broken regardless of distribution | More fundamental problem — LoRA may not be enough; revisit training loop |
| Zero-shot + finetuning dist significantly worse than zero-shot + county own | Expected: county-specific context is more informative | Confirms geo-pooling's per-county context design is correct |
| Finetuned + finetuning dist ≈ zero-shot + finetuning dist (both ~50) | Finetuning added nothing, ICL is carrying all the signal | Suggests the finetuning is basically a no-op for performance |

**Also note** `y_swap mean` vs `y_train mean` in the output header. If they're far apart (e.g., swap mean=11.62 vs county mean=10.20), the distribution mismatch is large, which is expected to hurt the finetuned model under county-specific context.

---

## Decision tree after running both diagnostics

```
Tier 1: Shuffled-y MAPE ≈ Normal MAPE for finetuned?
│
├── YES (< 10% change) → ICL is broken
│   │
│   └── Tier 2: Finetuning-dist context helps?
│       ├── YES → Distribution-specific ICL collapse
│       │         → Try LoRA (preserves base weights)
│       │         → Or: finetune on more diverse data
│       └── NO  → Total ICL collapse
│                 → LoRA alone may not suffice
│                 → Consider: smaller LR, fewer epochs,
│                   mixing synthetic tasks into finetuning
│
└── NO (> 30% MAPE increase with shuffled-y) → ICL still works
    │
    └── Check predictions: are finetuned preds near global prior?
        ├── YES → Systematic bias (county-distribution mismatch)
        │         → Try county-level y normalization at inference
        └── NO  → Some other issue — inspect predictions manually
```
