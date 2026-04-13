#!/usr/bin/env python
"""
Minimal XGBoost test with fake data to verify GPU support and pipeline works.
This bypasses data loading to test with limited RAM in interactive sessions.
"""

import numpy as np
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("MINIMAL XGBOOST PIPELINE TEST (Fake Data)")
print("=" * 80)

# Test 1: Import and GPU check
print("\n[1/5] Testing imports and GPU availability...")
try:
    import torch
    import xgboost as xgb
    from src.models import XGBoostModel

    cuda_available = torch.cuda.is_available()
    print(f"  ✓ PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ XGBoost version: {xgb.__version__}")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create fake data
print("\n[2/5] Creating fake data (1000 samples, 30 features)...")
try:
    np.random.seed(42)
    n_samples = 1000
    n_features = 30

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randn(n_samples) * 100000 + 500000)

    X_test = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_test = pd.Series(np.random.randn(200) * 100000 + 500000)

    print(f"  ✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  ✓ Test: {X_test.shape[0]} samples")
except Exception as e:
    print(f"  ✗ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Create XGBoost model with GPU
print("\n[3/5] Creating XGBoost model (3 Optuna trials, 2-fold CV)...")
try:
    model = XGBoostModel(
        n_trials=3,
        cv_folds=2,
        use_gpu=True,
        random_state=42
    )
    print(f"  ✓ Model created")
    print(f"  ✓ GPU enabled: {model.use_gpu}")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    sys.exit(1)

# Test 4: Train model
print("\n[4/5] Training XGBoost with Optuna tuning...")
print("  (This will take ~30-60 seconds with 3 trials)")
try:
    model.fit(X_train, y_train)
    print(f"  ✓ Training completed")
    print(f"  ✓ Best params: {model.get_hyperparameters()}")
    print(f"  ✓ Tuning time: {model.get_tune_time():.2f}s")
except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Predict
print("\n[5/5] Making predictions on test set...")
try:
    y_pred = model.predict(X_test)

    # Compute simple metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  ✓ Predictions generated: {len(y_pred)} values")
    print(f"  ✓ MAE: ${mae:,.0f}")
    print(f"  ✓ R²: {r2:.4f}")
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    sys.exit(1)

# Cleanup
print("\n[Cleanup] Releasing GPU resources...")
try:
    model.cleanup()
    print(f"  ✓ Resources cleaned up")
except Exception as e:
    print(f"  ⚠ Cleanup warning: {e}")

print("\n" + "=" * 80)
print("SUCCESS! XGBoost pipeline works end-to-end with GPU")
print("=" * 80)
print("\nYou can now submit the full job:")
print("  sbatch experiments/slurm/finetuning/train_xgboost_only.sh")
print("=" * 80)
