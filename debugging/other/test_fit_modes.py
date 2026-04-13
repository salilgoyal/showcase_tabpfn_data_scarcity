"""
Test TabPFN fit_mode performance with different configurations.

This script compares:
1. Default mode (no fit_mode specified)
2. fit_preprocessors mode
3. batched mode (and what happens when predict is called)

Usage:
    # On an interactive node with GPU:
    sdev --partition=gpu --gpus=1
    cd /home/users/salilg/tabpfn_data_scarcity
    source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
    python debugging/test_fit_modes.py
"""

import numpy as np
import pandas as pd
import time
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data similar to the experiment
# Cross-county experiment: 10K train, 358K test, 107 features
N_TRAIN = 10_000
N_TEST = 5_000  # Reduced for 5.81 GiB GPU memory constraints
N_FEATURES = 107
BATCH_SIZE = 1000  # Batch size for prediction to avoid OOM

print("=" * 80)
print("TabPFN Fit Mode Performance Test")
print("=" * 80)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Train samples: {N_TRAIN:,}")
print(f"Test samples: {N_TEST:,}")
print(f"Features: {N_FEATURES}")
print()

# Generate synthetic data
print("Generating synthetic data...")
X_train = pd.DataFrame(
    np.random.randn(N_TRAIN, N_FEATURES),
    columns=[f"feat_{i}" for i in range(N_FEATURES)]
)
y_train = pd.Series(np.random.randn(N_TRAIN) * 2 + 10)

X_test = pd.DataFrame(
    np.random.randn(N_TEST, N_FEATURES),
    columns=[f"feat_{i}" for i in range(N_FEATURES)]
)
y_test = pd.Series(np.random.randn(N_TEST) * 2 + 10)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test configurations
test_configs = [
    {
        "name": "Default mode (no fit_mode)",
        "fit_mode": None,
    },
    {
        "name": "fit_preprocessors mode",
        "fit_mode": "fit_preprocessors",
    },
    {
        "name": "batched mode",
        "fit_mode": "batched",
    },
]

results = []

for config in test_configs:
    print("=" * 80)
    print(f"Testing: {config['name']}")
    print("=" * 80)

    try:
        # Create model
        print("Creating model...")
        kwargs = {
            "device": device,
            "ignore_pretraining_limits": True,
        }
        if config["fit_mode"] is not None:
            kwargs["fit_mode"] = config["fit_mode"]

        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            **kwargs
        )

        print(f"Model created with fit_mode: {model.fit_mode}")

        # Fit
        print(f"Fitting on {len(X_train):,} samples...")
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start
        print(f"  Fit time: {fit_time:.2f}s")
        print(f"  Model fit_mode after fit: {model.fit_mode}")

        # Predict in batches to avoid OOM
        print(f"Predicting on {len(X_test):,} samples in batches of {BATCH_SIZE}...")
        pred_start = time.time()

        # Predict in batches
        y_pred_list = []
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(X_test))
            X_batch = X_test.iloc[i:batch_end]
            y_pred_batch = model.predict(X_batch)
            y_pred_list.append(y_pred_batch)

            # Clear cache after each batch
            if device == 'cuda' and i + BATCH_SIZE < len(X_test):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        y_pred = np.concatenate(y_pred_list)
        pred_time = time.time() - pred_start
        print(f"  Predict time: {pred_time:.2f}s")
        print(f"  Model fit_mode after predict: {model.fit_mode}")

        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        print(f"  MSE: {mse:.4f}")

        results.append({
            "config": config["name"],
            "fit_mode": config["fit_mode"],
            "fit_time": fit_time,
            "pred_time": pred_time,
            "total_time": fit_time + pred_time,
            "mse": mse,
            "status": "success"
        })

        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

        results.append({
            "config": config["name"],
            "fit_mode": config["fit_mode"],
            "fit_time": None,
            "pred_time": None,
            "total_time": None,
            "mse": None,
            "status": f"failed: {str(e)}"
        })

    print()

# Print summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Analyze prediction time
print("=" * 80)
print("PREDICTION TIME ANALYSIS")
print("=" * 80)
successful_results = [r for r in results if r["status"] == "success" and r["pred_time"] is not None]
if len(successful_results) > 1:
    fastest = min(successful_results, key=lambda x: x["pred_time"])
    slowest = max(successful_results, key=lambda x: x["pred_time"])

    print(f"Fastest: {fastest['config']} - {fastest['pred_time']:.2f}s")
    print(f"Slowest: {slowest['config']} - {slowest['pred_time']:.2f}s")
    print(f"Slowdown: {slowest['pred_time'] / fastest['pred_time']:.2f}x")
    print()

    # Extrapolate to full test set (358K samples from cross-county experiment)
    full_test_samples = 358_000
    full_test_ratio = full_test_samples / N_TEST
    print(f"Extrapolated to {full_test_samples:,} test samples (cross-county experiment size):")
    print(f"  Fastest: {fastest['pred_time'] * full_test_ratio / 60:.2f} minutes")
    print(f"  Slowest: {slowest['pred_time'] * full_test_ratio / 60:.2f} minutes")
    print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
if successful_results:
    best = min(successful_results, key=lambda x: x["pred_time"])
    print(f"Best fit_mode for non-finetuning use case: {best['fit_mode'] or 'default'}")
    print(f"  Prediction time: {best['pred_time']:.2f}s")
    print()

    # Check if batched mode caused issues
    batched_result = next((r for r in results if r["fit_mode"] == "batched"), None)
    if batched_result and batched_result["status"] == "success":
        if batched_result["pred_time"] > best["pred_time"] * 2:
            print("WARNING: 'batched' mode is significantly slower for prediction!")
            print(f"  batched prediction: {batched_result['pred_time']:.2f}s")
            print(f"  best prediction: {best['pred_time']:.2f}s")
            print(f"  slowdown: {batched_result['pred_time'] / best['pred_time']:.2f}x")
            print()
            print("'batched' mode is designed for fine-tuning, not standard in-context learning.")
            print("TabPFN automatically switches mode during predict(), which may cause slowdown.")
