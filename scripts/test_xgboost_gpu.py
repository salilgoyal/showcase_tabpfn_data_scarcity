#!/usr/bin/env python
"""
Quick test to verify XGBoost GPU support on Sherlock.

Run this on a GPU node to check if XGBoost can use CUDA.
"""

import sys
import numpy as np
import pandas as pd

print("=" * 80)
print("XGBOOST GPU SUPPORT TEST")
print("=" * 80)

# Check CUDA availability via PyTorch
try:
    import torch
    print(f"\n1. PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - Device: {torch.cuda.get_device_name(0)}")
        print(f"   - Device count: {torch.cuda.device_count()}")
except ImportError:
    print("\n1. PyTorch not installed")

# Check XGBoost installation
try:
    import xgboost as xgb
    print(f"\n2. XGBoost version: {xgb.__version__}")
    print(f"   - XGBoost path: {xgb.__file__}")
except ImportError:
    print("\n2. XGBoost not installed!")
    sys.exit(1)

# Test GPU histogram method
print("\n3. Testing GPU support...")
try:
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.randn(100))

    # Try to create model with GPU
    model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        device='cuda',
        n_estimators=10
    )

    print("   - Created XGBRegressor with tree_method='gpu_hist'")

    # Try to fit
    model.fit(X, y)
    print("   ✓ GPU training SUCCESSFUL!")

    # Make prediction
    pred = model.predict(X[:5])
    print(f"   ✓ GPU prediction works (sample: {pred[0]:.4f})")

except Exception as e:
    print(f"   ✗ GPU training FAILED: {e}")
    print("\n   Trying CPU fallback...")
    try:
        model_cpu = xgb.XGBRegressor(
            tree_method='hist',
            n_estimators=10
        )
        model_cpu.fit(X, y)
        print("   ✓ CPU training works")
    except Exception as e2:
        print(f"   ✗ CPU training also failed: {e2}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Final recommendation
try:
    import xgboost as xgb
    model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=1)
    X_test = pd.DataFrame(np.random.randn(10, 5))
    y_test = pd.Series(np.random.randn(10))
    model.fit(X_test, y_test)
    print("✓ XGBoost GPU is WORKING on this system")
    print("  You can use use_gpu: true in your configs")
except:
    print("✗ XGBoost GPU is NOT working")
    print("\nPossible fixes:")
    print("  1. Reinstall XGBoost with GPU support:")
    print("     pip uninstall xgboost")
    print("     pip install xgboost")
    print("     (Recent XGBoost versions have GPU support by default)")
    print("\n  2. Or install from conda-forge:")
    print("     conda install -c conda-forge py-xgboost-gpu")
    print("\n  3. Set use_gpu: false in your config to use CPU")

print("=" * 80)
