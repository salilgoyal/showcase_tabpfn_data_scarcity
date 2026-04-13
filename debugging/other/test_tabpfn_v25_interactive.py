#!/usr/bin/env python
"""
Interactive debug script for TabPFN v2.5 testing.

This script replicates the exact flow of cross_county.py but with a small dataset
to quickly identify issues without waiting for batch jobs.

Usage:
    # Get interactive GPU node first:
    srun -p gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash

    # Then run this script:
    module load python/3.12 cuda
    source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
    export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
    python debugging/test_tabpfn_v25_interactive.py
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CleanedDataLoader
from src.data.split_strategies import load_test_set_result, load_train_set_result, get_train_test_data
from src.data.preprocessing_utils import apply_phase2_preprocessing
from src.models.tabpfn_wrapper import TabPFNModel
from src.evaluation import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Test TabPFN v2.5 with the exact same flow as cross_county experiment."""

    logger.info("=" * 80)
    logger.info("TabPFN v2.5 Interactive Debug Test")
    logger.info("=" * 80)

    # Configuration - mirrors cross_county experiment
    config = {
        'data': {
            'cleaned_data_path': '/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/',
            'target_column': 'SALE_AMOUNT'
        },
        'splits': {
            'test_set_dir': '/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/',
            'train_set_dir': '/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v4/'
        },
        'preprocessing': {
            'phase2_steps': {
                'winsorize': True,
                'winsorize_percentile': 1,
                'normalize_continuous': True,
                'impute_method': 'median'
            }
        },
        'tabpfn': {
            'version': 'v2.5',  # Testing v2.5
            'device': 'cuda'
        },
        'experiment': {
            'random_seed': 42
        }
    }

    # OPTION 1: Small subset for quick testing (recommended for initial test)
    SMALL_TEST = True
    MAX_TRAIN = 1000 if SMALL_TEST else 5000   # Start small, increase if it works
    MAX_TEST = 500 if SMALL_TEST else 2000

    logger.info(f"Test mode: {'SMALL' if SMALL_TEST else 'MEDIUM'}")
    logger.info(f"Max train samples: {MAX_TRAIN}, Max test samples: {MAX_TEST}")
    logger.info("")

    # Step 1: Load data (same as cross_county.py)
    logger.info("Step 1: Loading data...")
    data_loader = CleanedDataLoader(
        cleaned_data_path=config['data']['cleaned_data_path'],
        target_column=config['data']['target_column'],
        phase2_config={}  # Don't apply Phase 2 in loader
    )

    # Load test/train splits (same as cross_county.py)
    logger.info("Loading pre-generated test set...")
    test_result = load_test_set_result(config['splits']['test_set_dir'])
    logger.info(f"  Test counties: {len(test_result.test_counties)}")
    logger.info(f"  Test samples: {len(test_result.test_indices)}")

    logger.info("Loading pre-generated train set...")
    train_result = load_train_set_result(config['splits']['train_set_dir'])
    logger.info(f"  Train samples: {len(train_result.train_indices)}")
    logger.info(f"  Source breakdown: {train_result.source_breakdown}")
    logger.info("")

    # Subsample for quick testing
    logger.info(f"Subsampling to {MAX_TRAIN} train and {MAX_TEST} test samples for quick test...")
    train_indices_subset = train_result.train_indices[:MAX_TRAIN]
    test_indices_subset = test_result.test_indices[:MAX_TEST]

    # Load only the needed rows
    logger.info("Loading data...")
    all_indices = np.concatenate([test_indices_subset, train_indices_subset])
    unique_indices = np.unique(all_indices)
    df = data_loader.load_data_by_indices(unique_indices)
    logger.info(f"  Loaded {len(df):,} rows")

    # Remap indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
    test_indices_remapped = np.array([index_map[idx] for idx in test_indices_subset])
    train_indices_remapped = np.array([index_map[idx] for idx in train_indices_subset])
    df = df.reset_index(drop=True)

    # Step 2: Get train/test data (same as cross_county.py)
    logger.info("\nStep 2: Preparing train/test split...")
    test_result.test_indices = test_indices_remapped
    train_result.train_indices = train_indices_remapped

    X_train, y_train, X_test, y_test = get_train_test_data(
        df=df,
        test_result=test_result,
        train_result=train_result,
        target_column=config['data']['target_column'],
        exclude_columns=None
    )
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info("")

    # Step 3: Apply Phase 2 preprocessing (same as cross_county.py)
    logger.info("Step 3: Applying Phase 2 preprocessing...")
    X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config['preprocessing']['phase2_steps']
    )
    logger.info(f"  After Phase 2: {X_train.shape[1]} features")
    logger.info("")

    # Step 4: Create and fit TabPFN model (same as cross_county.py via base.py)
    logger.info("Step 4: Creating TabPFN v2.5 model...")
    logger.info("=" * 80)
    model = TabPFNModel(
        device=config['tabpfn']['device'],
        version=config['tabpfn']['version'],
        random_state=config['experiment']['random_seed']
    )
    logger.info(f"TabPFN model created: version={model.version}, device={model.device}")
    logger.info("")

    # Step 5: Fit model
    logger.info("Step 5: Fitting TabPFN model...")
    logger.info("=" * 80)
    logger.info("Starting fit()... (this is where the hang might occur)")
    sys.stdout.flush()
    sys.stderr.flush()

    import time
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    logger.info(f"✓ Fit completed in {fit_time:.2f} seconds ({fit_time/60:.2f} minutes)")
    logger.info("")

    # Step 6: Predict
    logger.info("Step 6: Making predictions...")
    logger.info("=" * 80)
    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - pred_start

    logger.info(f"✓ Prediction completed in {pred_time:.2f} seconds")
    logger.info("")

    # Step 7: Compute metrics
    logger.info("Step 7: Computing metrics...")
    metrics = compute_metrics(y_test.values, y_pred, log_transformed=True)

    logger.info("=" * 80)
    logger.info("SUCCESS! TabPFN v2.5 works correctly")
    logger.info("=" * 80)
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")
    logger.info(f"Metrics: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. If this worked, try increasing MAX_TRAIN to 5000, then 10000")
    logger.info("  2. Monitor timing - if fit() takes >30min for 10K samples, it's too slow")
    logger.info("  3. If it hangs at larger sizes, we need to investigate TabPFN v2.5 fit mode")

    # Cleanup
    model.cleanup()
    logger.info("\nCleanup complete")

if __name__ == '__main__':
    main()
