"""
Test TabPFN performance with a sample of real property tax data.
This will help determine if the slowdown is data-dependent.
"""

import numpy as np
import pandas as pd
import time
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/users/salilg/tabpfn_data_scarcity')

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from src.data.loading import CleanedDataLoader
from src.data.preprocessing_utils import Phase2Preprocessor

print("=" * 80)
print("Testing TabPFN with Real Property Tax Data Sample")
print("=" * 80)

# Load a small sample of real data
DATA_FILE = "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet"
SAMPLE_SIZE = 20_000  # Take 20K samples total

print(f"Loading sample of {SAMPLE_SIZE:,} rows from real data...")
print(f"Data file: {DATA_FILE}")

try:
    # Load a random sample
    df = pd.read_parquet(DATA_FILE)
    print(f"Full dataset: {len(df):,} rows")

    # Sample
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled: {len(df_sample):,} rows")

    # Split into train/test
    train_size = 10_000
    test_size = SAMPLE_SIZE - train_size

    # Separate features and target
    TARGET_COL = 'SALE_AMOUNT'
    y = df_sample[TARGET_COL]
    X = df_sample.drop(columns=[TARGET_COL])

    # Log transform target
    y = np.log1p(y)

    print(f"Features: {X.shape[1]}")
    print(f"Train: {train_size:,} samples")
    print(f"Test: {test_size:,} samples")

    # Split
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Apply Phase 2 preprocessing (like in the experiment)
    print("\nApplying Phase 2 preprocessing...")
    preprocessor = Phase2Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"After preprocessing: {X_train_processed.shape[1]} features")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test 1: fit_preprocessors mode
    print("\n" + "=" * 80)
    print("TEST 1: fit_preprocessors mode")
    print("=" * 80)

    model1 = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        device=device,
        fit_mode="fit_preprocessors",
        ignore_pretraining_limits=True
    )

    print("Fitting...")
    t0 = time.time()
    model1.fit(X_train_processed, y_train)
    t1 = time.time()
    fit_time_1 = t1 - t0
    print(f"  Fit time: {fit_time_1:.2f}s")

    print("Predicting...")
    t0 = time.time()
    y_pred1 = model1.predict(X_test_processed)
    t1 = time.time()
    pred_time_1 = t1 - t0
    print(f"  Predict time: {pred_time_1:.2f}s")
    print(f"  Total: {fit_time_1 + pred_time_1:.2f}s")

    del model1
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Test 2: batched mode
    print("\n" + "=" * 80)
    print("TEST 2: batched mode (may auto-switch)")
    print("=" * 80)

    model2 = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        device=device,
        fit_mode="batched",
        ignore_pretraining_limits=True
    )

    print("Fitting...")
    t0 = time.time()
    model2.fit(X_train_processed, y_train)
    t1 = time.time()
    fit_time_2 = t1 - t0
    print(f"  Fit time: {fit_time_2:.2f}s")

    print("Predicting...")
    t0 = time.time()
    y_pred2 = model2.predict(X_test_processed)
    t1 = time.time()
    pred_time_2 = t1 - t0
    print(f"  Predict time: {pred_time_2:.2f}s")
    print(f"  Total: {fit_time_2 + pred_time_2:.2f}s")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH REAL DATA")
    print("=" * 80)
    print(f"fit_preprocessors: {fit_time_1 + pred_time_1:.2f}s total")
    print(f"batched mode: {fit_time_2 + pred_time_2:.2f}s total")

    if (fit_time_2 + pred_time_2) > (fit_time_1 + pred_time_1) * 1.5:
        slowdown = (fit_time_2 + pred_time_2) / (fit_time_1 + pred_time_1)
        print(f"\n⚠️  batched mode is {slowdown:.1f}x slower with real data!")
        print("This explains the cross-county slowdown.")
    else:
        print("\n✓ No significant slowdown with real data")

    print("\nExtrapolating to cross-county sizes (10K train, 358K test):")
    test_scale = 358_000 / test_size
    print(f"  fit_preprocessors: {(fit_time_1 + pred_time_1 * test_scale) / 60:.1f} minutes")
    print(f"  batched mode: {(fit_time_2 + pred_time_2 * test_scale) / 60:.1f} minutes")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
