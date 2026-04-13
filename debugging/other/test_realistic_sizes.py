"""
Test with realistic data sizes to match cross-county experiment.
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

# Realistic sizes matching cross-county experiment
N_TRAIN = 10_000
N_TEST = 10_000  # Start with 10K test (can't do 358K due to memory)
N_FEATURES = 107

print("=" * 80)
print("Testing TabPFN with Cross-County Experiment Sizes")
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

print(f"Created with fit_mode: {model1.fit_mode}")

print("Fitting...")
t0 = time.time()
model1.fit(X_train, y_train)
t1 = time.time()
fit_time_1 = t1 - t0
print(f"  Fit time: {fit_time_1:.2f}s")
print(f"  After fit, fit_mode: {model1.fit_mode}")

print("Predicting...")
t0 = time.time()
y_pred1 = model1.predict(X_test)
t1 = time.time()
pred_time_1 = t1 - t0
print(f"  Predict time: {pred_time_1:.2f}s")
print(f"  After predict, fit_mode: {model1.fit_mode}")

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

print(f"Created with fit_mode: {model2.fit_mode}")

print("Fitting...")
t0 = time.time()
model2.fit(X_train, y_train)
t1 = time.time()
fit_time_2 = t1 - t0
print(f"  Fit time: {fit_time_2:.2f}s")
print(f"  After fit, fit_mode: {model2.fit_mode}")

print("Predicting...")
t0 = time.time()
y_pred2 = model2.predict(X_test)
t1 = time.time()
pred_time_2 = t1 - t0
print(f"  Predict time: {pred_time_2:.2f}s")
print(f"  After predict, fit_mode: {model2.fit_mode}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"fit_preprocessors mode:")
print(f"  Fit: {fit_time_1:.2f}s")
print(f"  Predict: {pred_time_1:.2f}s")
print(f"  Total: {fit_time_1 + pred_time_1:.2f}s")
print()
print(f"batched mode:")
print(f"  Fit: {fit_time_2:.2f}s")
print(f"  Predict: {pred_time_2:.2f}s")
print(f"  Total: {fit_time_2 + pred_time_2:.2f}s")
print()

if fit_time_2 > fit_time_1 * 1.5:
    print(f"⚠️  batched mode fit is {fit_time_2/fit_time_1:.1f}x slower!")
elif pred_time_2 > pred_time_1 * 1.5:
    print(f"⚠️  batched mode predict is {pred_time_2/pred_time_1:.1f}x slower!")
else:
    print("✓ No significant slowdown detected with these data sizes")

print()
print("Extrapolating to cross-county sizes (358K test samples):")
test_scale = 358_000 / N_TEST
print(f"  fit_preprocessors total: {(fit_time_1 + pred_time_1 * test_scale) / 60:.1f} minutes")
print(f"  batched mode total: {(fit_time_2 + pred_time_2 * test_scale) / 60:.1f} minutes")
