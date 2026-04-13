# Baseline Comparison Guide

This guide explains how to compare your model predictions against county baseline assessments (CALCULATED_TOTAL_VALUE).

## Overview

The implementation now:
1. Keeps `CALCULATED_TOTAL_VALUE` in preprocessed data
2. Automatically excludes it from training features
3. Saves baseline values and adjustment ratios during train/test set generation
4. **NEW: Automatically evaluates baseline in experiments** - just add `baseline: {enabled: true}` to your config
5. Provides utilities for manual comparison (if needed)

## Changes Made

### 1. Preprocessing Configuration
- **File**: `preprocessing/configs/v1_no_onehot.yaml`
- **Change**: Set `assessed_value: true` to keep CALCULATED_TOTAL_VALUE in preprocessed data
- **Note**: The column is automatically excluded from training features

### 2. Split Strategies Module
- **File**: `src/data/split_strategies.py`
- **Changes**:
  - `save_test_set_result()` now saves:
    - `test_baseline_values.npy` - baseline values for test set
    - `test_sale_amounts.npy` - actual sale amounts for test set
    - `train_pool_baseline_values.npy` - baseline values for train pool
    - `train_pool_sale_amounts.npy` - sale amounts for train pool

  - `save_train_set_result()` now saves:
    - `train_baseline_values.npy` - baseline values for training set
    - `train_sale_amounts.npy` - sale amounts for training set
    - Computes and saves `baseline_adjustment_ratio` in metadata.json

  - `get_train_test_data()` automatically excludes CALCULATED_TOTAL_VALUE from features

### 3. Generation Scripts
- **Files**: `experiments/scripts/generate_test_set.py`, `experiments/scripts/generate_train_set.py`
- **Change**: Now pass the full DataFrame to save functions for baseline extraction

## Directory Structure

After generating train/test sets, your directory structure will be:

```
preprocessed/v1_no_onehot/
├── data.parquet                    # Contains CALCULATED_TOTAL_VALUE
├── test_v1/
│   ├── test_indices.npy
│   ├── test_baseline_values.npy    # NEW: baseline predictions for test set
│   ├── test_sale_amounts.npy       # NEW: actual sale amounts for test set
│   ├── train_pool_indices.npy
│   ├── train_pool_baseline_values.npy  # NEW
│   ├── train_pool_sale_amounts.npy     # NEW
│   └── train_v2/
│       ├── train_indices.npy
│       ├── train_baseline_values.npy   # NEW: baseline for training data
│       ├── train_sale_amounts.npy      # NEW: sales for training data
│       └── metadata.json               # Contains baseline_adjustment_ratio
```

## Quick Start: Integrated Baseline Evaluation

**The easiest way** to compare against baseline is to enable it in your experiment config.

### 1. Enable Baseline in Config

Add this to your experiment config (e.g., `experiments/configs/cross_county/test_v1_train_v2.yaml`):

```yaml
# ==============================================================================
# BASELINE
# ==============================================================================
baseline:
  enabled: true  # Set to false to skip baseline comparison
```

### 2. Run Your Experiment

```bash
python experiments/run_experiment.py \
    --config experiments/configs/cross_county/test_v1_train_v2.yaml
```

### 3. Check Results

The `results.csv` file will now include baseline metrics alongside your models:

```csv
model,r2,mae,rmse,mape,adjustment_ratio,...
baseline,0.65,45000,67000,18.5,1.1234,...
tabpfn,0.72,38000,55000,15.2,,...
xgboost,0.75,35000,52000,14.8,,...
```

**That's it!** The baseline is automatically:
- Evaluated using the same test data as models
- Adjusted by the training-data-based ratio
- Included in per-county results (if enabled)
- Saved to the same results file

## Manual Baseline Comparison (Advanced)

If you need to evaluate baseline outside of experiments, use the utility functions.

### Using the Baseline Model Directly

```python
from src.models import BaselineModel, load_baseline_data
import numpy as np

# Load baseline data
test_baseline, train_baseline, adjustment_ratio = load_baseline_data(
    test_set_dir="/path/to/test_v1",
    train_set_dir="/path/to/test_v1/train_v2"
)

# Create and fit baseline model
model = BaselineModel(
    baseline_values_train=train_baseline,
    baseline_values_test=test_baseline,
    adjustment_ratio=adjustment_ratio  # Optional - will be computed if None
)

# Fit (computes ratio if not provided)
import pandas as pd
y_train = pd.Series(...)  # Your training targets
model.fit(pd.DataFrame(), y_train)

# Predict
y_pred = model.predict(pd.DataFrame())

# Evaluate
from src.evaluation import compute_metrics
y_test = pd.Series(...)  # Your test targets
metrics = compute_metrics(y_test.values, y_pred, log_transformed=False)

print(f"Baseline R²: {metrics['r2']:.4f}")
print(f"Baseline MAE: {metrics['mae']:.2f}")
print(f"Baseline RMSE: {metrics['rmse']:.2f}")
print(f"Baseline MAPE: {metrics['mape']:.2f}%")
```

### Low-Level Example (Manual Loading)

```python
import numpy as np
import json

# Load test baseline values and adjustment ratio
test_baseline = np.load('test_v1/test_baseline_values.npy')

# Load adjustment ratio from train set metadata
with open('test_v1/train_v2/metadata.json', 'r') as f:
    metadata = json.load(f)
    adjustment_ratio = metadata['baseline_adjustment_ratio']

# Get adjusted baseline predictions
adjusted_baseline_predictions = test_baseline * adjustment_ratio

# Load your model predictions
model_predictions = np.load('your_model_predictions.npy')

# Load actual sale amounts
test_sales = np.load('test_v1/test_sale_amounts.npy')

# Compare
from sklearn.metrics import mean_absolute_error, r2_score

baseline_mae = mean_absolute_error(test_sales, adjusted_baseline_predictions)
model_mae = mean_absolute_error(test_sales, model_predictions)

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Model MAE: {model_mae:.2f}")
print(f"Improvement: {(baseline_mae - model_mae) / baseline_mae * 100:.1f}%")
```

### Understanding the Adjustment Ratio

The `baseline_adjustment_ratio` (also called `adjustment_ratio` in the code) is computed as:
```python
median(SALE_AMOUNT / CALCULATED_TOTAL_VALUE)
```

This accounts for counties where assessed values are a fraction of market value (e.g., Cook County where assessments are 10% of market value).

**Important**: The ratio is computed only on the **training data** to avoid any test set leakage.

**Where it's stored:**
- Automatically saved in `train_set_dir/metadata.json` when you generate train sets
- Automatically loaded by the integrated baseline evaluation
- Can be manually loaded using `load_baseline_data()` function

### Using Both Raw and Adjusted Baselines

```python
# Compare against both raw and adjusted baselines
raw_baseline = test_baseline
adjusted_baseline = test_baseline * adjustment_ratio

# Evaluate both
raw_mae = mean_absolute_error(test_sales, raw_baseline)
adjusted_mae = mean_absolute_error(test_sales, adjusted_baseline)
model_mae = mean_absolute_error(test_sales, model_predictions)

print(f"Raw Baseline MAE: {raw_mae:.2f}")
print(f"Adjusted Baseline MAE: {adjusted_mae:.2f}")
print(f"Model MAE: {model_mae:.2f}")
```

## Regenerating Train/Test Sets

To regenerate your train/test sets with baseline values:

### 1. Rerun Preprocessing (if needed)
If you haven't already run preprocessing with `assessed_value: true`:

```bash
python preprocessing/scripts/clean_pooled_data.py \
    --config preprocessing/configs/v1_no_onehot.yaml
```

### 2. Regenerate Test Set
```bash
python experiments/scripts/generate_test_set.py \
    --config experiments/configs/test_sets/test_v1.yaml \
    --data_path /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet \
    --output_dir /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/
```

### 3. Regenerate Train Sets
```bash
python experiments/scripts/generate_train_set.py \
    --config experiments/configs/train_sets/train_v2.yaml \
    --test_split_dir /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/ \
    --data_path /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet \
    --output_dir /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v2/
```

Repeat for train_v5, train_v6, etc.

## How It Works

1. **Automatic Integration**: When you enable `baseline: {enabled: true}` in your config:
   - Experiment runner automatically loads baseline data after training models
   - Creates a `BaselineModel` that computes/uses the adjustment ratio
   - Evaluates baseline using the same test data as other models
   - Saves predictions to `predictions_baseline.parquet`
   - Includes baseline in per-county evaluation (if enabled)
   - Adds baseline row to `results.csv`

2. **No Memory Issues**: Baseline values are saved as separate small `.npy` files, so you never need to load the full `data.parquet` during evaluation.

3. **Automatic Exclusion**: `CALCULATED_TOTAL_VALUE` is automatically excluded from training features by `get_train_test_data()`, so you don't need to worry about it leaking into your models.

4. **Adjustment Ratio**: The ratio is computed per train set (not per county or globally), ensuring it's based only on training data and avoiding any test set leakage.

5. **Missing Values**: If baseline data files are missing, the experiment will log a warning and skip baseline evaluation (won't fail the entire experiment).

## Troubleshooting

**Q: Baseline is not appearing in my results**
- Check that `baseline: {enabled: true}` is in your experiment config
- Check experiment logs for warnings about missing baseline files
- Verify that `test_baseline_values.npy` exists in your test set directory
- Verify that `train_baseline_values.npy` and `adjustment_ratio` exist in your train set directory

**Q: I get "Test baseline values not found" error**
- You need to regenerate your test set with the updated code
- Run: `sbatch experiments/slurm/splits/generate_test_set.sh experiments/configs/test_sets/test_v1.yaml`

**Q: I get "Could not load baseline data" warning**
- Your train/test splits were generated before baseline support was added
- Regenerate both test and train sets (see "Regenerating Train/Test Sets" section)

**Q: I get "Column 'CALCULATED_TOTAL_VALUE' not found"**
- Run preprocessing again with `assessed_value: true` in the config
- This should have been done when you first set up baseline support

**Q: My baseline values are all zeros or NaN**
- Check that CALCULATED_TOTAL_VALUE was present in the raw county CSV files
- Check preprocessing logs for any dropped columns

**Q: The adjustment ratio seems wrong (too high/low)**
- This can happen if a county uses very different assessment practices
- Examine the distribution: `np.percentile(train_sales / train_baseline, [25, 50, 75])`
- Consider computing per-county ratios if counties have very different practices
- Check the `adjustment_ratio` value in `train_set_dir/metadata.json`

**Q: Can I use this with existing train/test splits?**
- No, you need to regenerate the splits with the updated code
- The old splits don't have the baseline value files

**Q: Can I disable baseline for specific experiments?**
- Yes, just set `baseline: {enabled: false}` or omit the `baseline` section entirely
