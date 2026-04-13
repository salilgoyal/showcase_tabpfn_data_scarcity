#!/usr/bin/env python
"""
Test TabPFN v2.5 with small subset of real data.
Loads just 2000 rows total to avoid memory issues.
"""

import sys
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("TabPFN v2.5 Test with Real Data (Small Subset)")
    logger.info("=" * 80)

    # Load small subset directly from parquet
    data_path = "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet"

    logger.info(f"Loading 2000 rows from {data_path}...")
    df = pd.read_parquet(data_path, engine='pyarrow')
    df = df.head(2000)  # Just first 2000 rows
    logger.info(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Simple train/test split
    target_col = 'SALE_AMOUNT'
    exclude_cols = ['fips', 'sale_date', 'SALE_AMOUNT', 'CALCULATED_TOTAL_VALUE']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    logger.info(f"Features: {len(feature_cols)}")

    X = df[feature_cols].fillna(0)  # Simple imputation
    y = np.log1p(df[target_col])  # Log transform

    # Split
    split_idx = 1500
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info("")

    # Test TabPFN v2.5
    logger.info("Creating TabPFN v2.5 model...")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5,
        device='cuda',
        fit_mode="batched",
        ignore_pretraining_limits=True
    )
    logger.info("✓ Model created")

    # Fit
    logger.info("Fitting model...")
    model.fit_mode = "fit_preprocessors"

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    logger.info(f"✓ Fit completed in {fit_time:.2f}s")

    # Predict
    logger.info("Predicting...")
    model.fit_mode = "batched"

    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - pred_start

    logger.info(f"✓ Prediction completed in {pred_time:.2f}s")

    # Metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info("=" * 80)
    logger.info("SUCCESS with Real Data!")
    logger.info("=" * 80)
    logger.info(f"Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")
    logger.info(f"R²: {r2:.4f}, MAE: {mae:.4f}")
    logger.info("")
    logger.info("Next: Check why full experiment data loading caused OOM kill")

if __name__ == '__main__':
    main()
