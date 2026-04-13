#!/usr/bin/env python
"""
Minimal TabPFN v2.5 test with synthetic data.
Tests if TabPFN v2.5 works at all without loading real data.
"""

import sys
import logging
import numpy as np
import pandas as pd
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("Minimal TabPFN v2.5 Test with Synthetic Data")
    logger.info("=" * 80)

    # Generate synthetic data
    n_train = 1000
    n_test = 500
    n_features = 50

    logger.info(f"Generating synthetic data: {n_train} train, {n_test} test, {n_features} features")

    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(n_train, n_features))
    y_train = pd.Series(np.random.randn(n_train))
    X_test = pd.DataFrame(np.random.randn(n_test, n_features))
    y_test = pd.Series(np.random.randn(n_test))

    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info("")

    # Test TabPFN v2.5
    logger.info("Testing TabPFN v2.5...")
    logger.info("=" * 80)

    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    logger.info("Creating TabPFN v2.5 model...")
    sys.stdout.flush()

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5,
        device='cuda',
        fit_mode="batched",
        ignore_pretraining_limits=True
    )
    logger.info("✓ Model created")
    logger.info("")

    # Switch to fit_preprocessors mode
    logger.info("Switching to fit_preprocessors mode...")
    model.fit_mode = "fit_preprocessors"
    logger.info("✓ Switched to fit_preprocessors mode")
    logger.info("")

    # Fit
    logger.info(f"Calling fit() on {n_train} samples...")
    sys.stdout.flush()

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    logger.info(f"✓ Fit completed in {fit_time:.2f} seconds")
    logger.info("")

    # Switch back to batched mode
    logger.info("Switching back to batched mode for inference...")
    model.fit_mode = "batched"
    logger.info("✓ Switched to batched mode")
    logger.info("")

    # Predict
    logger.info(f"Predicting on {n_test} samples...")
    sys.stdout.flush()

    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - pred_start

    logger.info(f"✓ Prediction completed in {pred_time:.2f} seconds")
    logger.info("")

    # Results
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info("=" * 80)
    logger.info("SUCCESS! TabPFN v2.5 works correctly")
    logger.info("=" * 80)
    logger.info(f"Train: {n_train} samples, Test: {n_test} samples, Features: {n_features}")
    logger.info(f"Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")
    logger.info(f"R²: {r2:.4f}, MAE: {mae:.4f}")
    logger.info("")
    logger.info("TabPFN v2.5 is working! The issue with the full experiment is elsewhere.")

if __name__ == '__main__':
    main()
