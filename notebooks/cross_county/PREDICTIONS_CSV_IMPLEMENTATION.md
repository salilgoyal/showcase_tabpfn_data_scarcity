# Test Predictions CSV Implementation

## Overview

Implemented functionality to save test set predictions to CSV files for the cross-county small in-context experiment. Each iteration now produces a CSV file with predictions from all models.

## Changes Made

### 1. Modified `create_sampled_split()` in [cross_county.py](../../experiments/experiment_types/cross_county.py:143-251)

**What changed:**
- Added tracking of FIPS codes for each test sample
- Return type changed from `Tuple[X, y, X, y]` to `Tuple[X, y, X, y, fips_array]`
- As we build the test set, we now track which county each row came from

**Key code:**
```python
test_fips_labels = []  # Track which FIPS each test row came from

# When adding test samples:
test_fips_labels.extend([allocation.fips] * len(test_indices))

# Return as numpy array:
return X_train, y_train, X_test, y_test, np.array(test_fips_labels)
```

### 2. Modified `run_single_iteration()` in [cross_county.py](../../experiments/experiment_types/cross_county.py:344-485)

**What changed:**
- Capture `test_fips_labels` from `create_sampled_split()`
- Collect predictions from each model during training
- Call `_save_test_predictions_csv()` after all models complete

**Key code:**
```python
# Capture FIPS labels
test_fips_labels = None
if self.sampling_result is not None:
    X_train, y_train, X_test, y_test, test_fips_labels = self.create_sampled_split(...)

# Collect predictions
model_predictions = {}
for model_name in enabled_models:
    # ... train model ...
    y_pred = model.predict(X_test)

    # Store predictions
    if test_fips_labels is not None:
        model_predictions[model_name] = y_pred

# Save CSV
if test_fips_labels is not None and len(model_predictions) > 0:
    self._save_test_predictions_csv(
        X_test, y_test, test_fips_labels, model_predictions, iteration
    )
```

### 3. Added `_save_test_predictions_csv()` method in [cross_county.py](../../experiments/experiment_types/cross_county.py:603-642)

**What it does:**
- Creates a DataFrame with FIPS, true values, and predictions from all models
- Includes all feature columns for detailed analysis
- Saves to `{output_dir}/test_predictions_iter{N}.csv`

**CSV structure:**
```
fips, y_true, tabpfn_pred, xgboost_pred, [feature1, feature2, ...]
```

## Output Files

When you run the experiment with [small_in_context_10k.yaml](../../experiments/configs/cross_county/small_in_context_10k.yaml):

```
/scratch/users/salilg/property_tax/results/cross_county/small_in_context_10k/
├── results.csv                      # Aggregate results (10 iterations × 2 models = 20 rows)
├── test_predictions_iter0.csv       # ~10K rows with predictions for iteration 0
├── test_predictions_iter1.csv       # ~10K rows with predictions for iteration 1
├── ...
└── test_predictions_iter9.csv       # ~10K rows with predictions for iteration 9
```

Each predictions CSV has:
- **fips**: County FIPS code for this row
- **y_true**: True sale amount (log-transformed)
- **tabpfn_pred**: TabPFN's prediction
- **xgboost_pred**: XGBoost's prediction
- **[features]**: All preprocessed features used for training

## Usage Example

After running the experiment, you can analyze per-county performance:

```python
import pandas as pd
import numpy as np

# Load predictions from one iteration
df = pd.read_csv('.../test_predictions_iter0.csv')

# Compute metrics per FIPS
from sklearn.metrics import r2_score, mean_absolute_error

results = []
for fips in df['fips'].unique():
    df_fips = df[df['fips'] == fips]

    # TabPFN metrics
    tabpfn_r2 = r2_score(df_fips['y_true'], df_fips['tabpfn_pred'])
    tabpfn_mae = mean_absolute_error(df_fips['y_true'], df_fips['tabpfn_pred'])

    # XGBoost metrics
    xgb_r2 = r2_score(df_fips['y_true'], df_fips['xgboost_pred'])
    xgb_mae = mean_absolute_error(df_fips['y_true'], df_fips['xgboost_pred'])

    results.append({
        'fips': fips,
        'n_samples': len(df_fips),
        'tabpfn_r2': tabpfn_r2,
        'tabpfn_mae': tabpfn_mae,
        'xgboost_r2': xgb_r2,
        'xgboost_mae': xgb_mae
    })

df_results = pd.DataFrame(results)
print(df_results.sort_values('tabpfn_r2'))
```

## Verification

Run the verification script to check the implementation:

```bash
conda activate tabpfn_env
python notebooks/cross_county/verify_implementation.py
```

All checks should pass:
- ✓ create_sampled_split returns test_fips_labels
- ✓ _save_test_predictions_csv method exists
- ✓ test_fips_labels being tracked and populated
- ✓ predictions being collected
- ✓ CSV saving method being called

## Notes

- CSV files are only generated for **sampled experiments** (using `SmallCountyInContextSampler`)
- Traditional cross-county experiments (without sampling) do NOT generate these CSVs
- File size: Each CSV is approximately 5-10 MB (10K rows × ~50 features)
- Total disk usage: ~50-100 MB for 10 iterations
