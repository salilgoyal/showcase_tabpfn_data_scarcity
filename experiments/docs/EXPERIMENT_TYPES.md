# Experiment Types and Execution Flow

This document provides detailed pseudocode for each experiment type, showing exactly how data is split, how models are trained, and how results are collected.

## Table of Contents

1. [Within-County Cross-Validation](#1-within-county-cross-validation)
2. [Cross-County Generalization](#2-cross-county-generalization)
3. [Data Scaling](#3-data-scaling)
4. [Comparison Summary](#comparison-summary)

---

## 1. Within-County Cross-Validation

**Purpose**: Evaluate model performance within a single county using repeated K-fold cross-validation.

**Key Characteristics**:
- Tests generalization **within** the same county
- Uses K-fold CV to maximize data usage
- Multiple repetitions reduce variance from fold assignment

### Pseudocode

```python
# ============================================
# WITHIN-COUNTY CROSS-VALIDATION
# ============================================

# Load and preprocess a single county
df = load_and_preprocess_county(fips)
X, y = split_features_target(df)

# For each REPETITION (different fold assignments)
for repeat in range(n_repeats):  # e.g., 10 repetitions
    
    # Create K folds with different random seed
    folds = create_k_folds(X, y, k=k_folds, random_state=base_seed + repeat)
    # e.g., k_folds=5 creates 5 train/test splits
    
    # For each FOLD (leave-one-out)
    for fold_idx in range(k_folds):  # e.g., 5 folds
        
        # ====================================
        # CREATE TRAIN/TEST SPLIT
        # ====================================
        X_train = concat([folds[i] for i in range(k_folds) if i != fold_idx])
        y_train = concat([folds[i] for i in range(k_folds) if i != fold_idx])
        
        X_test = folds[fold_idx]
        y_test = folds[fold_idx]
        
        # Example: 80% train, 20% test for k=5
        
        # ====================================
        # TRAIN AND EVALUATE EACH MODEL
        # ====================================
        for model_name in enabled_models:  # e.g., TabPFN, XGBoost
            
            # Create fresh model instance
            model = create_model(model_name)
            
            # FIT on K-1 folds
            model.fit(X_train, y_train)
            # Note: XGBoost does Optuna tuning internally with CV on X_train
            
            # PREDICT on held-out fold
            y_pred = model.predict(X_test)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            
            # Save result with metadata
            result = {
                'fips': fips,
                'fold': fold_idx,
                'repeat': repeat,
                'model': model_name,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X_train.shape[1],
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'fit_time': fit_time,
                'pred_time': pred_time,
                'status': 'success'
            }
            results.append(result)
            
            # Cleanup model
            model.cleanup()

# Save all results
save_results(results, f'county_{fips}_results.csv')
```

### Total Experiments
For a single county with **k=5 folds**, **10 repetitions**, **2 models**:
- **Total = 5 × 10 × 2 = 100 experiments** per county

### Example Config
```yaml
experiment:
  type: "within_county"
  repetitions: 10
  
# K-folds determined per county (e.g., 5 for large, 3 for small)
```

---

## 2. Cross-County Generalization

**Purpose**: Test how well models trained on data from multiple counties generalize to a held-out target county.

**Key Characteristics**:
- Tests generalization **across** different counties
- Uses pooled training data from all counties
- **NO K-fold CV** - single train/test split per iteration
- Uses feature intersection to handle mismatches

### Pseudocode

```python
# ============================================
# CROSS-COUNTY GENERALIZATION EXPERIMENT
# ============================================

# Load all counties once (with preprocessing)
county_data = {}  # {fips: (X, y)}
for fips in county_fips_list:  # e.g., 9 counties
    df = load_and_preprocess_county(fips)
    X, y = split_features_target(df)
    county_data[fips] = (X, y)

# For each county as the HELD-OUT TARGET
for target_fips in county_data.keys():  # Outer loop: 9 counties
    
    # For each RANDOM ITERATION (reduces variance)
    for iteration in range(iterations):  # Middle loop: 10 iterations
        
        # ====================================
        # CREATE TRAIN/TEST SPLIT
        # ====================================
        X_target, y_target = county_data[target_fips]
        
        # Sample TEST SET from target county only
        test_indices = random_sample(
            X_target, 
            size=max(min_test_samples, test_fraction * len(X_target))
        )  # 20% or at least 50 samples
        train_indices = remaining_indices(X_target, test_indices)
        
        X_test = X_target[test_indices]  # From target county only
        y_test = y_target[test_indices]
        
        # POOL TRAINING DATA from ALL counties
        X_train_pool = []
        y_train_pool = []
        
        for fips, (X, y) in county_data.items():
            if fips == target_fips:
                # Use REMAINING 80% of target county for training
                X_train_pool.append(X_target[train_indices])
                y_train_pool.append(y_target[train_indices])
            else:
                # Use ALL data from non-target counties
                X_train_pool.append(X)
                y_train_pool.append(y)
        
        # Concatenate all training data into single pool
        X_train = concat(X_train_pool)  # Pooled from all counties
        y_train = concat(y_train_pool)
        
        # ====================================
        # ALIGN FEATURES (handle mismatches)
        # ====================================
        # Different counties may have different features
        # (e.g., missing census variables, property types)
        
        common_features = set(X_train.columns) ∩ set(X_test.columns)
        X_train = X_train[common_features]  # Keep only common features
        X_test = X_test[common_features]
        
        # ====================================
        # TRAIN AND EVALUATE EACH MODEL
        # ====================================
        for model_name in enabled_models:  # Inner loop: TabPFN, XGBoost
            
            # Create fresh model instance
            model = create_model(model_name)
            
            # FIT on pooled training data (NO K-FOLD CV)
            model.fit(X_train, y_train)
            # Note: For XGBoost, Optuna tuning happens inside .fit()
            #       using internal CV on X_train
            
            # PREDICT on target county test set
            y_pred = model.predict(X_test)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            
            # Save result with metadata
            result = {
                'target_fips': target_fips,
                'iteration': iteration,
                'model': model_name,
                'train_size': len(X_train),  # From all counties
                'test_size': len(X_test),    # From target only
                'n_features': X_train.shape[1],
                'n_counties': len(county_data),
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'fit_time': fit_time,
                'pred_time': pred_time,
                'status': 'success'
            }
            results.append(result)
            
            # Cleanup model
            model.cleanup()

# Save all results
save_results(results, 'results.csv')
```

### Total Experiments
For **9 counties**, **10 iterations**, **2 models**:
- **Total = 9 × 10 × 2 = 180 experiments**

### Key Differences from Within-County
- ❌ **NO K-fold cross-validation** (single split per iteration)
- ✅ **Pooled training** from multiple counties
- ✅ **Feature intersection** for compatibility
- ✅ **Includes 80% of target** in training (not pure zero-shot)

### Example Config
```yaml
experiment:
  type: "cross_county"
  
county_fips_list: [1011, 1041, 1065, 1075, ...]

test_fraction: 0.2
min_test_samples: 50
iterations: 10
```

---

## 3. Data Scaling

**Purpose**: Study how model performance scales with training data size, using a fixed test set.

**Key Characteristics**:
- Tests learning curves with varying training sizes
- Uses single large county (e.g., Cook County)
- Fixed test set across all training sizes
- Multiple seeds for each training size

### Pseudocode

```python
# ============================================
# DATA SCALING EXPERIMENT
# ============================================

# Load and preprocess county
df = load_and_preprocess_county(fips)
X, y = split_features_target(df)

# ====================================
# CREATE FIXED TEST SET (once)
# ====================================
# Sample a large test set that will be used for ALL training sizes
test_size = min(max_test_size, int(test_fraction * len(X)))
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, 
    test_size=test_size,
    random_state=base_seed
)

# Define training set sizes to evaluate
train_sizes = [100, 500, 1000, 2000, 5000, 10000, ...]
# Or as fractions: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

# ====================================
# FOR EACH TRAINING SIZE
# ====================================
for train_size in train_sizes:
    
    # Ensure we have enough data
    if train_size > len(X_train_full):
        continue
    
    # ====================================
    # FOR EACH RANDOM SEED (multiple samples)
    # ====================================
    for seed_idx in range(n_seeds):  # e.g., 20 seeds
        
        # Sample training subset of specified size
        X_train_subset, y_train_subset = random_sample(
            X_train_full, 
            y_train_full,
            size=train_size,
            random_state=base_seed + seed_idx
        )
        
        # ====================================
        # TRAIN AND EVALUATE EACH MODEL
        # ====================================
        for model_name in enabled_models:  # e.g., TabPFN, XGBoost
            
            # Create fresh model instance
            model = create_model(model_name)
            
            # FIT on training subset
            model.fit(X_train_subset, y_train_subset)
            
            # PREDICT on FIXED test set
            y_pred = model.predict(X_test)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            
            # Save result with metadata
            result = {
                'fips': fips,
                'train_size': train_size,
                'seed': seed_idx,
                'model': model_name,
                'test_size': len(X_test),
                'n_features': X_train_subset.shape[1],
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'fit_time': fit_time,
                'pred_time': pred_time,
                'status': 'success'
            }
            results.append(result)
            
            # Cleanup model
            model.cleanup()

# Save all results
save_results(results, 'results.csv')
```

### Total Experiments
For **10 training sizes**, **20 seeds**, **2 models**:
- **Total = 10 × 20 × 2 = 400 experiments**

### Key Characteristics
- ✅ **Fixed test set** (same test data for all training sizes)
- ✅ **Multiple seeds** at each size (reduces sampling variance)
- ✅ **No CV** (single train/test split)
- ✅ **Subsample without replacement** from training pool

### Example Config
```yaml
experiment:
  type: "data_scaling"
  
data:
  county_fips: 17031  # Cook County
  
train_sizes: [100, 500, 1000, 2000, 5000, 10000, 20000]
n_seeds: 20
test_fraction: 0.2
max_test_size: 10000
```

---

## Comparison Summary

| Aspect | Within-County | Cross-County | Data Scaling |
|--------|--------------|--------------|--------------|
| **Goal** | Performance within county | Generalization across counties | Learning curves with data size |
| **Data Scope** | Single county | Multiple counties pooled | Single large county |
| **CV Strategy** | K-fold CV | No CV (single split) | No CV (single split) |
| **Test Set** | Rotates through K folds | 20% of target county | Fixed test set |
| **Training Data** | K-1 folds from same county | All counties (pooled) | Variable-sized subsets |
| **Iterations** | K folds × N repeats | N random splits | N seeds × M train sizes |
| **Feature Handling** | Same features | Feature intersection | Same features |
| **Typical Total** | 100 per county | 180 for 9 counties | 400 for 10 sizes |
| **Use Case** | Baseline performance | Geographic transfer | Data requirements |

## Experiment Selection Guide

**Use Within-County when**:
- Establishing baseline performance for a specific county
- Data is limited to one geographic region
- Want to maximize use of available data via CV

**Use Cross-County when**:
- Testing if models learn generalizable patterns
- Have multiple similar datasets (counties)
- Want to pool data for better performance

**Use Data Scaling when**:
- Studying data efficiency of different models
- Have one large dataset to subsample
- Want to understand minimum data requirements
- Comparing sample efficiency (e.g., TabPFN vs XGBoost)

## Implementation Details

### Model Creation
All experiments use the same model initialization:

```python
if model_name == 'tabpfn':
    model = TabPFNModel(
        device='cuda',
        random_state=random_seed
    )
elif model_name == 'xgboost':
    model = XGBoostModel(
        n_trials=50,
        cv_folds=3,
        use_gpu=True,
        random_state=random_seed
    )
```

### XGBoost Hyperparameter Tuning
- Happens **inside** `model.fit()` via Optuna
- Uses **internal CV** on the training data
- Independent of the experiment's CV structure
- Tuning happens for **every** train/test split

### Metrics Computation
All experiments compute the same metrics:
- **R²**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error

If target is log-transformed (`log_transform_target: true`), predictions are automatically inverse-transformed before computing metrics.

### Random Seeds
- **Base seed**: From config (`experiment.random_seed`)
- **Iteration-specific seed**: `base_seed + iteration`
- Ensures reproducibility across runs

## Running Experiments

See [SLURM documentation](../slurm/README.md) for details on running experiments on Sherlock cluster.

Quick reference:

```bash
# Within-county (array job, one county per task)
sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/within_county/full_preprocessing.yaml

# Cross-county (single job, processes all counties)
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/small_counties.yaml

# Data scaling (single job, one large county)
sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_with_preprocessing.yaml
```

## Related Documentation

- [Preprocessing Configuration](README.md)
- [SLURM Scripts](../slurm/README.md)
- [Configuration Examples](../configs/)
