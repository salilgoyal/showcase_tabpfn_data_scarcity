"""
Minimal test to understand what happens when batched mode auto-switches.
"""

import numpy as np
import pandas as pd
import time
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Small dataset for quick testing
N_TRAIN = 1000
N_TEST = 500
N_FEATURES = 50

print("=" * 80)
print("Testing TabPFN Batched Mode Auto-Switch Behavior")
print("=" * 80)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Train: {N_TRAIN:,} samples, {N_FEATURES} features")
print(f"Test: {N_TEST:,} samples")
print()

# Generate data
print("Generating synthetic data...")
X_train = pd.DataFrame(np.random.randn(N_TRAIN, N_FEATURES))
y_train = pd.Series(np.random.randn(N_TRAIN))
X_test = pd.DataFrame(np.random.randn(N_TEST, N_FEATURES))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test 1: fit_preprocessors mode (expected to work)
print("\n" + "=" * 80)
print("TEST 1: Using fit_preprocessors mode directly")
print("=" * 80)

model1 = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2,
    device=device,
    fit_mode="fit_preprocessors",
    ignore_pretraining_limits=True
)

print(f"Model created with fit_mode: {model1.fit_mode}")

print("Fitting...")
t0 = time.time()
model1.fit(X_train, y_train)
t1 = time.time()
print(f"  Fit time: {t1-t0:.2f}s")
print(f"  Model fit_mode after fit: {model1.fit_mode}")

print("Predicting...")
t0 = time.time()
y_pred1 = model1.predict(X_test)
t1 = time.time()
print(f"  Predict time: {t1-t0:.2f}s")
print(f"  Model fit_mode after predict: {model1.fit_mode}")
print(f"  Predictions shape: {y_pred1.shape}")

del model1
if device == 'cuda':
    torch.cuda.empty_cache()

# Test 2: batched mode (will auto-switch during predict)
print("\n" + "=" * 80)
print("TEST 2: Using batched mode (will auto-switch during predict)")
print("=" * 80)

model2 = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2,
    device=device,
    fit_mode="batched",
    ignore_pretraining_limits=True
)

print(f"Model created with fit_mode: {model2.fit_mode}")

print("Fitting...")
t0 = time.time()
model2.fit(X_train, y_train)
t1 = time.time()
print(f"  Fit time: {t1-t0:.2f}s")
print(f"  Model fit_mode after fit: {model2.fit_mode}")

print("Predicting (this may auto-switch modes and be slow)...")
print("  Watching for warning message about mode switching...")
t0 = time.time()
y_pred2 = model2.predict(X_test)
t1 = time.time()
print(f"  Predict time: {t1-t0:.2f}s")
print(f"  Model fit_mode after predict: {model2.fit_mode}")
print(f"  Predictions shape: {y_pred2.shape}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"TEST 1 (fit_preprocessors): Predict took {t1-t0:.2f}s")
print(f"TEST 2 (batched->auto-switch): Predict took {t1-t0:.2f}s")
print()
print("Note: If TEST 2 is significantly slower, that explains the cross-county slowdown.")
