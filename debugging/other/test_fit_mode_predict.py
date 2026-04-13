import numpy as np
import pandas as pd
import time
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

np.random.seed(42)
torch.manual_seed(42)

# Small test
N_TRAIN = 2000
N_TEST = 2000
N_FEATURES = 50

print("Creating test data...")
X_train = pd.DataFrame(np.random.randn(N_TRAIN, N_FEATURES))
y_train = pd.Series(np.random.randn(N_TRAIN))
X_test = pd.DataFrame(np.random.randn(N_TEST, N_FEATURES))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test 1: fit_preprocessors mode (cross-county style)
print("\n" + "="*60)
print("TEST 1: fit_preprocessors mode (like cross-county)")
print("="*60)

model1 = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2,
    device=device,
    fit_mode="fit_preprocessors",
    ignore_pretraining_limits=True
)

t0 = time.time()
model1.fit(X_train, y_train)
fit_time1 = time.time() - t0

t0 = time.time()
y_pred1 = model1.predict(X_test)
pred_time1 = time.time() - t0

print(f"Fit: {fit_time1:.2f}s, Predict: {pred_time1:.2f}s, Total: {fit_time1+pred_time1:.2f}s")

del model1
if device == 'cuda':
    torch.cuda.empty_cache()
    time.sleep(2)

# Test 2: batched mode then switch back (finetuning style)
print("\n" + "="*60)
print("TEST 2: batched mode (like finetuning)")
print("="*60)

model2 = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2,
    device=device,
    fit_mode="batched",
    ignore_pretraining_limits=True
)

# Temporarily switch to fit_preprocessors to call fit()
model2.fit_mode = "fit_preprocessors"

t0 = time.time()
model2.fit(X_train, y_train)
fit_time2 = time.time() - t0

# Switch back to batched mode (like finetuning does)
model2.fit_mode = "batched"

t0 = time.time()
y_pred2 = model2.predict(X_test)
pred_time2 = time.time() - t0

print(f"Fit: {fit_time2:.2f}s, Predict: {pred_time2:.2f}s, Total: {fit_time2+pred_time2:.2f}s")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"fit_preprocessors: {pred_time1:.2f}s prediction")
print(f"batched: {pred_time2:.2f}s prediction")
if pred_time1 > pred_time2 * 1.1:
    print(f"\n⚠️ fit_preprocessors is {pred_time1/pred_time2:.1f}x SLOWER!")
elif pred_time2 > pred_time1 * 1.1:
    print(f"\n⚠️ batched is {pred_time2/pred_time1:.1f}x SLOWER!")
else:
    print("\n✓ Similar performance")
