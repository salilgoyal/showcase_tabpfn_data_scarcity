"""
Experiment 2: Cross-county generalization with pooled training.
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CountyDataLoader, PooledDataSplitter
from models import TabPFNModel, XGBoostModel
from evaluation import compute_metrics

logger = logging.getLogger(__name__)


class CrossCountyRunner:
    """Runner for cross-county pooled training experiment."""

    def __init__(self, config: Dict):
        """
        Initialize runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Extract preprocessing config
        preprocessing_config = config.get('preprocessing', {})

        # Check if new format (has 'features' and 'steps' keys) or old format
        if 'features' in preprocessing_config and 'steps' in preprocessing_config:
            # New format - use directly
            full_preprocessing_config = preprocessing_config
            logger.info("Using new modular preprocessing config format")
        elif 'use_evelyn_preprocessing' in preprocessing_config:
            # Old format - will be converted by CountyDataLoader
            logger.info("Using old preprocessing config format (will be auto-converted)")
            full_preprocessing_config = None
            # Pass old flags for backward compatibility
            use_evelyn = preprocessing_config.get('use_evelyn_preprocessing', False)
            include_chars = preprocessing_config.get('include_property_chars', False)
            chars_only = preprocessing_config.get('property_chars_only', False)

            self.data_loader = CountyDataLoader(
                county_csvs_dir=config['data']['county_csvs_dir'],
                target_column=config['data']['target_column'],
                use_evelyn_preprocessing=use_evelyn,
                include_property_chars=include_chars,
                property_chars_only=chars_only
            )

            # Store flag for passing to metrics computation
            self.log_transformed = use_evelyn
            return  # Early return for old format
        else:
            # No preprocessing
            full_preprocessing_config = None
            logger.info("No preprocessing configured")

        # New format initialization
        self.data_loader = CountyDataLoader(
            county_csvs_dir=config['data']['county_csvs_dir'],
            target_column=config['data']['target_column'],
            preprocessing_config=full_preprocessing_config
        )

        # Store flag for passing to metrics computation
        self.log_transformed = self.data_loader.is_log_transformed()

        if full_preprocessing_config:
            enabled_features = [k for k, v in full_preprocessing_config.get('features', {}).items() if v]
            logger.info(
                f"Using modular preprocessing with features: {', '.join(enabled_features)}"
            )

    def run_single_iteration(
        self,
        target_fips: int,
        all_fips_list: List[int],
        iteration: int,
        bin_name: str
    ) -> pd.DataFrame:
        """
        Run one iteration of pooled experiment for a target county.

        Args:
            target_fips: FIPS code of county to test on
            all_fips_list: List of all county FIPS codes in the pool
            iteration: Iteration number
            bin_name: Name of the size bin

        Returns:
            DataFrame with results for this iteration
        """
        logger.info(
            f"Starting iteration {iteration} for target county {target_fips}"
        )

        # Load all counties
        logger.info(f"Loading {len(all_fips_list)} counties...")
        county_data_dict = {}

        for fips in all_fips_list:
            try:
                df = self.data_loader.load_county(fips, drop_missing_target=True)
                X, y = self.data_loader.preprocess_for_training(df)
                county_data_dict[fips] = (X, y)
            except Exception as e:
                logger.error(f"Error loading county {fips}: {e}")

        if target_fips not in county_data_dict:
            raise ValueError(f"Target county {target_fips} not loaded successfully")

        # Create splitter
        splitter = PooledDataSplitter(
            test_fraction=self.config['test_sampling']['test_fraction'],
            min_test_samples=self.config['test_sampling']['min_test_samples'],
            n_iterations=self.config['test_sampling']['iterations'],
            random_state=self.config['experiment']['random_seed']
        )

        # Create train/test split
        X_train_pool, y_train_pool, X_test, y_test = splitter.create_pooled_split(
            county_data_dict=county_data_dict,
            target_county_fips=target_fips,
            iteration=iteration
        )

        logger.info(
            f"Split created: train_pool_size={len(X_train_pool)}, "
            f"test_size={len(X_test)}"
        )

        # Align features (in case some counties have different features)
        # Use intersection of features
        train_features = set(X_train_pool.columns)
        test_features = set(X_test.columns)
        common_features = list(train_features & test_features)

        # Log feature mismatches if they exist
        if train_features != test_features:
            missing_in_test = train_features - test_features
            missing_in_train = test_features - train_features
            logger.warning(
                f"Feature mismatch detected! "
                f"Missing in test: {len(missing_in_test)}, "
                f"Missing in train: {len(missing_in_train)}"
            )

        X_train_pool = X_train_pool[common_features]
        X_test = X_test[common_features]

        logger.info(f"Using {len(common_features)} common features")

        # Get model configs
        model_configs = self.config['models']
        enabled_models = [m['name'] for m in model_configs if m['enabled']]

        # Calibration storage (if enabled)
        calibration_enabled = self.config.get('calibration', {}).get('enabled', False)
        calibration_data = [] if calibration_enabled else None

        # Filter models for calibration experiments
        if calibration_enabled:
            calibration_models = self.config.get('calibration', {}).get('models', ['tabpfn'])
            enabled_models = [m for m in enabled_models if m in calibration_models]
            logger.info(f"Calibration mode: only running models {enabled_models}")

        # Train and evaluate each model
        results = []

        for model_name in enabled_models:
            try:
                result, cal_data = self._run_model_on_pooled_split(
                    model_name=model_name,
                    X_train=X_train_pool,
                    y_train=y_train_pool,
                    X_test=X_test,
                    y_test=y_test,
                    target_fips=target_fips,
                    bin_name=bin_name,
                    iteration=iteration
                )
                results.append(result)

                # Store calibration data if available
                if cal_data is not None:
                    calibration_data.append(cal_data)

            except Exception as e:
                logger.error(
                    f"Error running {model_name} on target county {target_fips}, "
                    f"iteration {iteration}: {e}",
                    exc_info=True
                )

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        logger.info(
            f"Completed iteration {iteration} for county {target_fips}: "
            f"{len(results_df)} results"
        )

        return results_df, calibration_data

    def _run_model_on_pooled_split(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_fips: int,
        bin_name: str,
        iteration: int
    ) -> Dict:
        """
        Train and evaluate a model on pooled data.

        Args:
            model_name: Name of model ('tabpfn' or 'xgboost')
            X_train, y_train: Pooled training data
            X_test, y_test: Test data from target county
            target_fips: Target county FIPS code
            bin_name: Size bin name
            iteration: Iteration number

        Returns:
            Tuple of (result dict, calibration data dict or None)
        """
        logger.debug(f"  Training {model_name} on pooled data...")

        # Create model
        if model_name == 'tabpfn':
            model = TabPFNModel(
                device='cuda',
                random_state=self.config['experiment']['random_seed']
            )
        elif model_name == 'xgboost':
            model = XGBoostModel(
                n_trials=self.config['xgboost']['optuna_trials'],
                cv_folds=self.config['xgboost']['optuna_cv_folds'],
                use_gpu=self.config['xgboost']['use_gpu'],
                random_state=self.config['experiment']['random_seed']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Fit model
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start

        # Predict
        pred_start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start

        # Calibration predictions (if enabled)
        calibration_data = None
        calibration_enabled = self.config.get('calibration', {}).get('enabled', False)
        if calibration_enabled and model_name == 'tabpfn':
            quantiles = self.config['calibration']['quantiles']
            logger.debug(f"  Predicting quantiles for calibration: {quantiles}")
            quantile_preds = model.predict_quantiles(X_test, quantiles)

            calibration_data = {
                'target_fips': target_fips,
                'model': model_name,
                'iteration': iteration,
                'y_true': y_test.values,
                'y_pred_mean': y_pred,
                'quantile_predictions': quantile_preds
            }

        # Compute metrics (pass log_transformed flag if using Evelyn preprocessing)
        metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)

        # Build result dictionary
        result = {
            'experiment_name': self.config.get('experiment', {}).get('name', 'default'),
            'experiment_description': self.config.get('experiment', {}).get('description', ''),
            'target_fips': target_fips,
            'bin_name': bin_name,
            'iteration': iteration,
            'model': model_name,
            'train_pool_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[1],
            'fit_time': fit_time,
            'pred_time': pred_time,
            **metrics
        }

        # Add hyperparameters if available
        if self.config['logging']['log_hyperparameters']:
            hyperparams = model.get_hyperparameters()
            if hyperparams and model_name == 'xgboost':
                # Flatten hyperparams with prefix
                for key, value in hyperparams.items():
                    result[f'hyperparam_{key}'] = value

                # Add tuning time for XGBoost
                if hasattr(model, 'get_tune_time'):
                    result['tune_time'] = model.get_tune_time()

        # Cleanup
        model.cleanup()

        logger.debug(
            f"  {model_name}: R2={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.2f}, fit_time={fit_time:.2f}s"
        )

        return result, calibration_data


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dicts, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_configs(base_config_path: Path, exp_config_path: Path, experiment_config_path: Path = None) -> dict:
    """
    Load and merge config files.

    Args:
        base_config_path: Path to base_config.yaml
        exp_config_path: Path to {within,cross}_county_config.yaml
        experiment_config_path: Optional path to experiment variant config

    Returns:
        Merged config dict
    """
    import yaml

    # Load base
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Load experiment-specific
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
    config.update(exp_config)

    # Load experiment variant (overrides)
    if experiment_config_path:
        logger.info(f"Loading experiment config: {experiment_config_path}")
        with open(experiment_config_path) as f:
            experiment_config = yaml.safe_load(f)

        # Deep merge (preserve nested dicts)
        config = deep_merge(config, experiment_config)

    return config


def main():
    """Main entry point for cross-county experiment."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Run cross-county experiment for one (county, iteration) pair'
    )
    parser.add_argument(
        '--target_fips',
        type=int,
        required=True,
        help='Target county FIPS code (to test on)'
    )
    parser.add_argument(
        '--fips_list',
        type=str,
        required=True,
        help='Comma-separated list of all county FIPS codes in the pool'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        help='Iteration number (0 to n_iterations-1)'
    )
    parser.add_argument(
        '--bin_name',
        type=str,
        required=True,
        help='Size bin name (e.g., "small")'
    )
    parser.add_argument(
        '--config_base',
        type=str,
        default='../config/base_config.yaml',
        help='Path to base config file'
    )
    parser.add_argument(
        '--config_experiment',
        type=str,
        default='../config/cross_county_config.yaml',
        help='Path to experiment config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--experiment_config',
        type=str,
        default=None,
        help='Path to experiment variant config (overrides base config)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse FIPS list
    all_fips_list = [int(x.strip()) for x in args.fips_list.split(',')]

    # Load configs
    config_dir = Path(__file__).parent.parent / 'config'
    base_config_path = config_dir / 'base_config.yaml'
    exp_config_path = config_dir / 'cross_county_config.yaml'

    # Handle experiment config path (relative to project root)
    experiment_config_path = None
    if args.experiment_config:
        # Get project root (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        experiment_config_path = project_root / args.experiment_config

    # Load and merge configs
    config = load_configs(
        base_config_path=base_config_path,
        exp_config_path=exp_config_path,
        experiment_config_path=experiment_config_path
    )

    # Override output dir if specified
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir

    # Handle output path templating
    experiment_name = config.get('experiment', {}).get('name', 'default')
    experiment_type = 'cross_county'

    output_dir = config['output']['results_dir']
    if '{experiment_type}' in output_dir or '{experiment_name}' in output_dir:
        output_dir = output_dir.format(
            experiment_type=experiment_type,
            experiment_name=experiment_name
        )
        config['output']['results_dir'] = output_dir
        logger.info(f"Output directory (after templating): {output_dir}")

    # Run experiment
    runner = CrossCountyRunner(config)

    results_df, calibration_data = runner.run_single_iteration(
        target_fips=args.target_fips,
        all_fips_list=all_fips_list,
        iteration=args.iteration,
        bin_name=args.bin_name
    )

    # Save results
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"county_{args.target_fips}_iter_{args.iteration}_results.csv"
    results_df.to_csv(output_file, index=False)

    logger.info(f"Results saved to {output_file}")

    # Save calibration data if available
    if calibration_data:
        import pickle

        calibration_file = output_dir / f"county_{args.target_fips}_iter_{args.iteration}_calibration.pkl"

        # Organize calibration data
        cal_output = {
            'target_fips': args.target_fips,
            'experiment_name': config.get('experiment', {}).get('name'),
            'quantiles': config['calibration']['quantiles'],
            'iterations': calibration_data
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(cal_output, f)

        logger.info(f"Calibration data saved to {calibration_file}")


if __name__ == '__main__':
    main()
