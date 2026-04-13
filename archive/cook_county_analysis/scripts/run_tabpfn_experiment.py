#!/usr/bin/env python3
"""
TabPFN Data Scarcity Experiment with CBG-matched test sets.

Evaluates TabPFN on:
1. Full test set (all 2022 data)
2. CBG-matched test set (only 2022 data from CBGs seen in training)

Outputs:
- results/tabpfn_full.csv
- results/tabpfn_cbg_matched.csv
- /nlp/scr/salilg/cook_county_predictions/tabpfn/...
"""

#SBATCH --account=nlp
#SBATCH --job-name=tabpfn_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=jag-standard
#SBATCH --time=4-0
#SBATCH --output=logs/tabpfn_experiment.out
#SBATCH --error=logs/tabpfn_experiment.err

import pandas as pd
import numpy as np
import logging
import sys
import os
import time
import argparse
import torch

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ====================================================================
# PREPROCESSING PIPELINE: Set via config file or defaults below
# ====================================================================
# If --config is provided, preprocessing will be set from config.
# Otherwise, defaults below are used (backward compatibility).

# Default preprocessing (used if no config file)
from src.evelyn_preprocessing_propertyonly import load_and_prepare_data_propertyonly_evelyn as load_and_prepare_data
USE_LOG_TRANSFORMED = True

# These will be overridden if --config is provided
_CONFIG = None
_LOAD_AND_PREPARE_DATA_FUNC = load_and_prepare_data
_USE_LOG_TRANSFORMED_FLAG = USE_LOG_TRANSFORMED

# ====================================================================

from src.data_utils import (
    sample_train_set,
    get_train_cbgs,
    subset_test_by_cbg,
    prepare_features_for_model
)
from src.evaluation import calculate_metrics, create_result_dict
from src.model_runners import TabPFNRunner

import warnings
warnings.filterwarnings('ignore')

# Logger will be configured in main() with experiment-specific path
logger = logging.getLogger(__name__)


def save_predictions(y_true, y_pred, X_test, seed, train_size, test_type,
                    predictions_dir, cbg_column, model_name='tabpfn'):
    """
    Save individual predictions to parquet file.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        X_test: Test features (with CBG column)
        seed: Random seed
        train_size: Training set size
        test_type: 'full' or 'cbg_matched'
        predictions_dir: Base directory for predictions
        cbg_column: Name of CBG column
        model_name: Name of model (default: 'tabpfn')
    """
    try:
        # predictions_dir already includes the model name, don't add it again
        os.makedirs(predictions_dir, exist_ok=True)

        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'y_true': y_true.values if hasattr(y_true, 'values') else y_true,
            'y_pred': y_pred,
            'cbg_id': X_test[cbg_column].values,
            'seed': seed,
            'train_size': train_size,
            'test_type': test_type
        }, index=y_true.index if hasattr(y_true, 'index') else range(len(y_true)))

        # Save to parquet with compression
        filename = f"seed{seed}_size{train_size}.parquet"
        filepath = os.path.join(predictions_dir, filename)
        predictions_df.to_parquet(filepath, compression='gzip', index=True)

        logger.debug(f"Saved predictions to {filepath}")

    except Exception as e:
        logger.warning(f"Failed to save predictions: {str(e)}")


def run_single_experiment(model_runner, X_train_pool, y_train_pool,
                          X_test_full, y_test_full,
                          train_size, seed, cbg_column, predictions_dir=None,
                          test_type='full'):
    """
    Run a single experiment with either full or CBG-matched test set.

    Args:
        model_runner: Instance of TabPFNRunner
        X_train_pool: Full training pool features
        y_train_pool: Full training pool targets
        X_test_full: Full test features
        y_test_full: Full test targets
        train_size: Number of training samples
        seed: Random seed
        cbg_column: Name of CBG column
        predictions_dir: Directory to save predictions (optional)
        test_type: 'full' or 'cbg_matched' (default: 'full')

    Returns:
        Result dictionary for the specified test type
    """
    logger.info(f"[Seed {seed}, Train Size {train_size}] Starting experiment (test_type={test_type})")

    try:
        # Sample training data
        X_train, y_train = sample_train_set(X_train_pool, y_train_pool, train_size, seed)
        n_train = len(X_train)

        # ====================================================================
        # CBG HANDLING: These functions may fail with Evelyn's preprocessing
        # because CBG column might not exist. Use try-except for robustness.
        # ====================================================================
        try:
            # Get training CBGs
            train_cbgs = get_train_cbgs(X_train, cbg_column)
            n_train_cbgs = len(train_cbgs)

            # Prepare features for modeling (drop CBG column)
            X_train_model = prepare_features_for_model(X_train, cbg_column, keep_cbg=False)
        except (KeyError, ValueError):
            # CBG column doesn't exist (e.g., with Evelyn's preprocessing)
            logger.warning(f"CBG column '{cbg_column}' not found - using full dataset")
            train_cbgs = set()
            n_train_cbgs = 0
            X_train_model = X_train

        # Determine which test set to use
        if test_type == 'cbg_matched':
            if train_cbgs:
                # Create CBG-matched test set
                X_test, y_test = subset_test_by_cbg(
                    X_test_full, y_test_full, train_cbgs, cbg_column
                )
                n_test_cbgs = len(set(X_test[cbg_column].dropna().unique()))
            else:
                logger.warning("Cannot create CBG-matched test set without CBG column - using full test set")
                X_test = X_test_full
                y_test = y_test_full
                n_test_cbgs = 0
        else:
            # Use full test set
            X_test = X_test_full
            y_test = y_test_full
            try:
                n_test_cbgs = len(set(X_test[cbg_column].dropna().unique()))
            except (KeyError, AttributeError):
                n_test_cbgs = 0

        try:
            X_test_model = prepare_features_for_model(X_test, cbg_column, keep_cbg=False)
        except (KeyError, ValueError):
            # CBG column doesn't exist
            X_test_model = X_test

        # Train model
        start_time = time.time()
        model_runner.fit(X_train_model, y_train)
        train_time = time.time() - start_time

        # Predict on test set
        start_time = time.time()
        y_pred = model_runner.predict(X_test_model)
        pred_time = time.time() - start_time

        # Calculate metrics (with log transformation if using Evelyn's preprocessing)
        use_log = _USE_LOG_TRANSFORMED_FLAG if _CONFIG is None else _CONFIG['preprocessing'].get('use_log_transform', False)
        metrics = calculate_metrics(y_test, y_pred, log_transformed=use_log)
        logger.info(f"[Seed {seed}, Train Size {train_size}] "
                   f"{test_type} MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")

        # Save predictions
        if predictions_dir:
            save_predictions(y_test, y_pred, X_test, seed, train_size,
                           test_type, predictions_dir, cbg_column, 'tabpfn')

        # Create result dictionary
        result = create_result_dict(
            seed, train_size, test_type, n_train, len(y_test),
            n_train_cbgs, n_test_cbgs,
            metrics, train_time, pred_time
        )

        return result

    except Exception as e:
        logger.error(f"[Seed {seed}, Train Size {train_size}] Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Return failed result
        result = create_result_dict(
            seed, train_size, test_type, 0, len(y_test_full),
            0, 0, {}, 0, 0, status=f'failed: {str(e)}'
        )

        return result

    finally:
        # ====================================================================
        # FIX: Guaranteed cleanup in finally block
        # ====================================================================
        model_runner.cleanup()
        logger.debug(f"[Seed {seed}, Train Size {train_size}] Cleanup complete")
        # ====================================================================


def run_full_experiment(experiment_name, data_path, output_dir='results',
                       train_sizes=None, seeds=None, cbg_column='block_group_id',
                       predictions_dir=None,
                       test_type='full'):
    """
    Run the full TabPFN data scarcity experiment.

    Args:
        experiment_name: Name for this experiment (required, used in output filenames)
        data_path: Path to cook_county.csv
        output_dir: Directory to save results
        train_sizes: List of training sizes (default: 100 to 1000 by 100)
        seeds: List of random seeds (default: 0-9)
        cbg_column: Name of CBG column
        predictions_dir: Directory to save predictions (default: /nlp/scr/salilg/cook_county_predictions)
        test_type: 'full' or 'cbg_matched' (default: 'full')
    """
    if train_sizes is None:
        train_sizes = [25, 50, 75] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
    if seeds is None:   
        seeds = [100 * i for i in range(20)]

    logger.info("="*80)
    logger.info(f"TABPFN DATA SCARCITY EXPERIMENT: {experiment_name}")
    logger.info("="*80)

    # Create output directories with experiment name
    exp_output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    if predictions_dir:
        predictions_dir_full = os.path.join(predictions_dir, experiment_name, 'tabpfn')
        os.makedirs(predictions_dir_full, exist_ok=True)
        logger.info(f"Predictions will be saved to: {predictions_dir_full}")
    else:
        predictions_dir_full = None

    # Load and prepare data (use config function if available, otherwise default)
    load_func = _LOAD_AND_PREPARE_DATA_FUNC if _CONFIG is None else _CONFIG.get('_load_func', load_and_prepare_data)
    X_train_pool, y_train_pool, X_test, y_test, cbg_column = load_func(
        data_path, cbg_column
    )

    # Initialize model runner (use CPU if CUDA not available)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model_runner = TabPFNRunner(device=device)

    # Experiment parameters
    total_experiments = len(train_sizes) * len(seeds)
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Test type: {test_type}")
    logger.info(f"Train sizes: {train_sizes}")
    logger.info(f"Seeds: {list(seeds)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Train pool size: {len(X_train_pool)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info("="*80)

    # Run experiments
    results = []
    experiment_num = 0

    start_time_total = time.time()

    for seed in seeds:
        for train_size in train_sizes:
            experiment_num += 1
            logger.info(f"\n--- Experiment {experiment_num}/{total_experiments} ---")

            result = run_single_experiment(
                model_runner, X_train_pool, y_train_pool,
                X_test, y_test,
                train_size, seed, cbg_column, predictions_dir_full,
                test_type=test_type
            )

            results.append(result)

            # Save intermediate results every 20 experiments
            if experiment_num % 20 == 0:
                df_results = pd.DataFrame(results)
                output_file = os.path.join(exp_output_dir, 'tabpfn.csv')
                df_results.to_csv(output_file, index=False)
                logger.info(f"Saved intermediate results to {output_file}")

    # Save final results
    df_results = pd.DataFrame(results)
    output_file = os.path.join(exp_output_dir, 'tabpfn.csv')
    df_results.to_csv(output_file, index=False)

    total_time = time.time() - start_time_total

    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_time/60:.2f} minutes")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {sum(r['status'] == 'success' for r in results)}")

    if predictions_dir_full:
        logger.info(f"Predictions saved to: {predictions_dir_full}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TabPFN Data Scarcity Experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment config YAML file (if provided, overrides other args)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment (used in output filenames)')
    parser.add_argument('--data_path', type=str,
                       default='../data/cook_county.csv',
                       help='Path to cook_county.csv')
    parser.add_argument('--output_dir', type=str,
                       default='../results',
                       help='Output directory for results')
    parser.add_argument('--cbg_column', type=str,
                       default='block_group_id',
                       help='Name of census block group column')
    parser.add_argument('--predictions_dir', type=str,
                       default='/oak/stanford/groups/deho/salilg/cook_county_predictions', #'/nlp/scr/salilg/cook_county_predictions',
                       help='Directory to save predictions')
    parser.add_argument('--no_predictions', action='store_true',
                       help='Skip saving predictions')
    parser.add_argument('--test_type', type=str,
                       choices=['full', 'cbg_matched'],
                       default='full',
                       help='Test set type: full or cbg_matched (default: full)')
    parser.add_argument('--train_sizes', nargs='+', type=int, default=None,
                       help='Training sizes to sweep (default: None)')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        from src.config_loader import load_config, setup_preprocessing_from_config, get_experiment_params, get_paths_config
        _CONFIG = load_config(args.config)
        _LOAD_AND_PREPARE_DATA_FUNC, _USE_LOG_TRANSFORMED_FLAG = setup_preprocessing_from_config(_CONFIG)
        _CONFIG['_load_func'] = _LOAD_AND_PREPARE_DATA_FUNC
        
        # Override args with config values
        if args.experiment_name is None:
            args.experiment_name = _CONFIG['experiment_name']
        paths = get_paths_config(_CONFIG)
        exp_params = get_experiment_params(_CONFIG)
        
        args.data_path = paths.get('data_path', args.data_path)
        args.output_dir = paths.get('output_dir', args.output_dir)
        args.cbg_column = exp_params.get('cbg_column', args.cbg_column)
        args.test_type = exp_params.get('test_type', args.test_type)
        if args.train_sizes is None:
            args.train_sizes = exp_params.get('train_sizes', None)
        
        # Handle predictions_dir
        if args.no_predictions:
            predictions_dir = None
        else:
            predictions_dir = paths.get('predictions_dir', args.predictions_dir)
            if predictions_dir is None:
                predictions_dir = None  # Explicitly None if not in config
    else:
        # Backward compatibility: require experiment_name if no config
        if args.experiment_name is None:
            parser.error("--experiment_name is required when --config is not provided")
        # Set predictions_dir to None if --no_predictions flag is set
        predictions_dir = None if args.no_predictions else args.predictions_dir

    # Update logging to include experiment name in subdirectory
    log_dir = os.path.join('..', 'logs', args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'tabpfn.log')
    # Reconfigure logging with experiment-specific filename
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    # Get seeds from config if available
    seeds = None
    if _CONFIG:
        exp_params = get_experiment_params(_CONFIG)
        seeds = exp_params.get('seeds', None)
    
    df_results = run_full_experiment(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        cbg_column=args.cbg_column,
        predictions_dir=predictions_dir,
        test_type=args.test_type,
        train_sizes=args.train_sizes,
        seeds=seeds
    )

    print("\n" + "="*80)
    print("TABPFN EXPERIMENT COMPLETE!")
    print(f"Experiment: {args.experiment_name}")
    print(f"Test type: {args.test_type}")
    print(f"Results saved to: {args.output_dir}/{args.experiment_name}/tabpfn.csv")
    print(f"Logs saved to: {log_dir}/tabpfn.log")
    if predictions_dir:
        print(f"Predictions saved to: {predictions_dir}/{args.experiment_name}/tabpfn/")
    print("="*80)
