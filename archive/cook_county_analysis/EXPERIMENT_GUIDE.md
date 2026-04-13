# Experiment Organization Guide

## Problem
As experiments accumulate, it becomes difficult to:
- Remember what each experiment tested
- Compare results across experiments
- Avoid duplicating work
- Reproduce past results

## Standard ML Engineering Practices

### 1. **Structured Naming Convention** ⭐ EASIEST TO IMPLEMENT

Use a consistent format: `<feature_set>-<preprocessing>-<n_seeds>seeds`

**Examples:**
```
results/
├── minimal-evelyn-20seeds/           # 21 features, Evelyn preprocessing
├── propertyonly-original-20seeds/    # Property chars only, no preprocessing
├── fullfeatures-evelyn-20seeds/      # All features, Evelyn preprocessing
└── assessedonly-original-20seeds/    # Just assessed value + time
```

**Feature Set Abbreviations:**
- `minimal` = assessed value + census + time (21 features)
- `propertyonly` = char_* features only (~110)
- `fullfeatures` = everything (~131)
- `assessedonly` = CALCULATED_TOTAL_VALUE + time only
- `censusonly` = census demographics + time

**Preprocessing:**
- `original` = no transformations
- `evelyn` = log + winsorize + normalize

### 2. **Experiment Registry** ⭐ RECOMMENDED

Maintain `experiments.yaml` with:
- Unique ID (exp_001, exp_002, ...)
- Date, description, hypothesis
- Configuration (features, preprocessing, seeds)
- Results location
- Key findings
- Next steps

**Benefits:**
- One file shows all experiments at a glance
- Easy to search and compare
- Serves as lab notebook
- Can be version controlled

**Usage:**
```bash
# Before running experiment, add to experiments.yaml
# After completing, update with key findings
git add experiments.yaml results/
git commit -m "exp_003: property-only features baseline"
```

### 3. **Industry Tools** (For larger projects)

**MLflow** (Most popular):
```python
import mlflow

mlflow.set_experiment("cook-county-preprocessing")

with mlflow.start_run(run_name="evelyn-minimal"):
    mlflow.log_params({
        "preprocessing": "evelyn",
        "features": "minimal",
        "n_features": 21,
        "seeds": 20
    })

    # Run experiment
    results = run_experiment(...)

    mlflow.log_metrics({
        "mae_10k": results['mae'],
        "r2_10k": results['r2']
    })
    mlflow.log_artifact("results/evelyn-minimal-20seeds/xgboost.csv")
```

**Weights & Biases (W&B)**: Better visualization, real-time monitoring

**DVC**: Version control for data and models

### 4. **Hierarchical Organization**

Group by research question:

```
results/
├── 01_preprocessing_comparison/
│   ├── original-fullfeatures-40seeds/
│   └── evelyn-fullfeatures-20seeds/
├── 02_feature_ablation/
│   ├── minimal-evelyn-20seeds/
│   ├── propertyonly-evelyn-20seeds/
│   └── assessedonly-evelyn-20seeds/
├── 03_temporal_experiments/
│   ├── allyears-evelyn-20seeds/
│   └── 2021only-evelyn-20seeds/
└── archive/
    └── exploratory/
```

**Benefits:**
- Tells the story of your research
- Easy to write paper from this structure
- Clear what each experiment group tests

### 5. **Results Summary Script**

Create `scripts/summarize_experiments.py`:

```python
import pandas as pd
import glob

def summarize_experiment(exp_dir):
    """Extract key metrics from experiment results."""
    xgb = pd.read_csv(f"{exp_dir}/xgboost.csv")
    pfn = pd.read_csv(f"{exp_dir}/tabpfn.csv")

    return {
        'name': exp_dir.split('/')[-1],
        'xgb_mae_1k': xgb[xgb.train_size==1000]['mae'].median(),
        'xgb_mae_10k': xgb[xgb.train_size==10000]['mae'].median(),
        'pfn_mae_1k': pfn[pfn.train_size==1000]['mae'].median(),
        'pfn_mae_10k': pfn[pfn.train_size==10000]['mae'].median(),
    }

# Run on all experiments
exps = glob.glob("results/*/")
summary = pd.DataFrame([summarize_experiment(e) for e in exps])
summary.to_csv("EXPERIMENT_SUMMARY.csv")
print(summary.sort_values('xgb_mae_10k'))
```

## Recommendation for Your Project

**Immediate steps:**

1. ✅ **Use `experiments.yaml`** (already created for you)
   - Update it before/after each experiment
   - Add key findings and next steps

2. ✅ **Adopt naming convention**
   - Rename existing experiments:
     ```bash
     mv results/evelyn-nopropertychars-20-seeds results/minimal-evelyn-20seeds
     mv results/evelyn-includepropertychars-20-seeds results/fullfeatures-evelyn-20seeds
     ```

3. ✅ **Use submission scripts**
   ```bash
   # For sequential execution (safer, easier to debug)
   sbatch scripts/submit_both_experiments.sh

   # For parallel execution (faster)
   bash scripts/submit_both_parallel.sh propertyonly-original-20seeds
   ```

4. 📝 **Create experiment summary**
   - After each experiment, update `experiments.yaml`
   - Run comparison notebook to visualize results

5. 🗂️ **Archive old experiments**
   ```bash
   mkdir -p results/archive/exploratory
   mv results/10_31_experiments results/archive/exploratory/
   mv results/11_3_experiments results/archive/exploratory/
   ```

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ Before starting new experiment:                         │
├─────────────────────────────────────────────────────────┤
│ 1. Add entry to experiments.yaml                        │
│ 2. Create output directory: results/<exp_name>/         │
│ 3. Document hypothesis and config                       │
│                                                          │
│ During experiment:                                       │
├─────────────────────────────────────────────────────────┤
│ 1. Monitor: tail -f outfiles/<exp_name>/*.out          │
│ 2. Check: squeue -u $USER                              │
│                                                          │
│ After experiment:                                        │
├─────────────────────────────────────────────────────────┤
│ 1. Update experiments.yaml with key findings            │
│ 2. Compare results in notebook                          │
│ 3. Document next steps                                   │
│ 4. Commit to git if important                           │
└─────────────────────────────────────────────────────────┘
```

## Git Workflow (Optional but Recommended)

```bash
# Before experiment
git checkout -b experiment/propertyonly-features
vim experiments.yaml  # Add experiment config
git commit -m "Plan exp_003: property-only features"

# After experiment
git add results/propertyonly-original-20seeds/
vim experiments.yaml  # Add key findings
git commit -m "Complete exp_003: MAE improved by 15%"
git push

# If successful, merge to main
git checkout main
git merge experiment/propertyonly-features
```

This creates a clear history of your experimental process!
