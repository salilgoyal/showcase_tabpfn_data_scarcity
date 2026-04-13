"""
Experiment 1: Within-county nested CV with per-fold hyperparameter tuning.
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

from data import CountyDataLoader, RepeatedKFoldSplitter
from models import TabPFNModel, XGBoostModel
from evaluation import compute_metrics

logger = logging.getLogger(__name__)


class WithinCountyRunner:
    """Runner for within-county nested CV experiment."""

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

    def run_county(
        self,
        fips: int,
        bin_name: str,
        k_folds: int,
        n_repeats: int
    ) -> pd.DataFrame:
        """
        Run experiment for a single county.

        Args:
            fips: County FIPS code
            bin_name: Name of the size bin
            k_folds: Number of folds
            n_repeats: Number of repetitions

        Returns:
            Tuple of (DataFrame with fold-level results, calibration data list or None, predictions data list or None)
        """
        logger.info(f"Starting experiment for county {fips} (bin: {bin_name})")

        # Load and preprocess data
        df = self.data_loader.load_county(fips, drop_missing_target=True)
        X, y = self.data_loader.preprocess_for_training(df)

        logger.info(
            f"County {fips}: {len(X)} samples, {X.shape[1]} features"
        )

        # Create splitter
        splitter = RepeatedKFoldSplitter(
            n_splits=k_folds,
            n_repeats=n_repeats,
            random_state=self.config['experiment']['random_seed']
        )

        # Results storage
        results = []

        # Calibration storage (if enabled)
        calibration_enabled = self.config.get('calibration', {}).get('enabled', False)
        calibration_data = [] if calibration_enabled else None

        # MODIFIED: Predictions storage (if enabled)
        predictions_enabled = self.config.get('predictions', {}).get('save_predictions', False)
        predictions_data = [] if predictions_enabled else None

        # Get model configs
        model_configs = self.config['models']
        enabled_models = [m['name'] for m in model_configs if m['enabled']]

        # Filter models for calibration experiments
        if calibration_enabled:
            calibration_models = self.config.get('calibration', {}).get('models', ['tabpfn'])
            enabled_models = [m for m in enabled_models if m in calibration_models]
            logger.info(f"Calibration mode: only running models {enabled_models}")

        # Nested CV loop
        total_splits = splitter.get_total_splits()
        logger.info(f"Running {total_splits} train/test splits...")

        for rep, fold, train_idx, test_idx in splitter.split(len(X)):
            split_num = rep * k_folds + fold + 1
            logger.info(
                f"County {fips} - Split {split_num}/{total_splits} "
                f"(rep={rep}, fold={fold})"
            )

            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            logger.debug(
                f"  Train size: {len(X_train)}, Test size: {len(X_test)}"
            )

            # Train and evaluate each model
            for model_name in enabled_models:
                try:
                    # MODIFIED: Now captures 3 return values (result, cal_data, pred_data)
                    result, cal_data, pred_data = self._run_model_on_fold(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        fips=fips,
                        bin_name=bin_name,
                        repetition=rep,
                        fold=fold
                    )
                    results.append(result)

                    # Store calibration data if available
                    if cal_data is not None:
                        calibration_data.append(cal_data)

                    # MODIFIED: Store prediction data if available
                    if pred_data is not None:
                        predictions_data.append(pred_data)

                except Exception as e:
                    logger.error(
                        f"Error running {model_name} on county {fips}, "
                        f"rep {rep}, fold {fold}: {e}",
                        exc_info=True
                    )

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        logger.info(f"Completed county {fips}: {len(results_df)} results")

        # MODIFIED: Return predictions_data alongside other outputs
        return results_df, calibration_data, predictions_data

    def _run_model_on_fold(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fips: int,
        bin_name: str,
        repetition: int,
        fold: int
    ) -> Dict:
        """
        Train and evaluate a model on one fold.

        Args:
            model_name: Name of model ('tabpfn' or 'xgboost')
            X_train, y_train: Training data
            X_test, y_test: Test data
            fips: County FIPS code
            bin_name: Size bin name
            repetition: Repetition number
            fold: Fold number

        Returns:
            Tuple of (result dict, calibration data dict or None, prediction data dict or None)
        """
        logger.debug(f"  Training {model_name}...")

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
                'fips': fips,
                'model': model_name,
                'repetition': repetition,
                'fold': fold,
                'y_true': y_test.values,
                'y_pred_mean': y_pred,
                'quantile_predictions': quantile_preds
            }

        # MODIFIED: Individual predictions storage (if enabled)
        prediction_data = None
        predictions_enabled = self.config.get('predictions', {}).get('save_predictions', False)
        if predictions_enabled:
            prediction_data = {
                'fips': fips,
                'model': model_name,
                'repetition': repetition,
                'fold': fold,
                'test_indices': y_test.index.values,  # Original indices for joining
                'y_true': y_test.values,
                'y_pred': y_pred
            }

        # Compute metrics (pass log_transformed flag if using Evelyn preprocessing)
        metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)

        # Build result dictionary
        result = {
            'experiment_name': self.config.get('experiment', {}).get('name', 'default'),
            'experiment_description': self.config.get('experiment', {}).get('description', ''),
            'fips': fips,
            'bin_name': bin_name,
            'repetition': repetition,
            'fold': fold,
            'model': model_name,
            'train_size': len(X_train),
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

        # MODIFIED: Return prediction data alongside result and calibration data
        return result, calibration_data, prediction_data


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
    """Main entry point for within-county experiment."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Run within-county experiment for a single county'
    )
    parser.add_argument(
        '--fips',
        type=int,
        required=True,
        help='County FIPS code'
    )
    parser.add_argument(
        '--bin_name',
        type=str,
        required=True,
        help='Size bin name (e.g., "small")'
    )
    parser.add_argument(
        '--k_folds',
        type=int,
        required=True,
        help='Number of CV folds'
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
        default='../config/within_county_config.yaml',
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

    # Load configs
    config_dir = Path(__file__).parent.parent / 'config'
    base_config_path = config_dir / 'base_config.yaml'
    exp_config_path = config_dir / 'within_county_config.yaml'

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
    experiment_type = 'within_county'

    output_dir = config['output']['results_dir']
    if '{experiment_type}' in output_dir or '{experiment_name}' in output_dir:
        output_dir = output_dir.format(
            experiment_type=experiment_type,
            experiment_name=experiment_name
        )
        config['output']['results_dir'] = output_dir
        logger.info(f"Output directory (after templating): {output_dir}")

    # Run experiment
    runner = WithinCountyRunner(config)

    # MODIFIED: Now captures 3 return values (results_df, calibration_data, predictions_data)
    results_df, calibration_data, predictions_data = runner.run_county(
        fips=args.fips,
        bin_name=args.bin_name,
        k_folds=args.k_folds,
        n_repeats=config['experiment']['repetitions']
    )

    # Save results
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"county_{args.fips}_results.csv"
    results_df.to_csv(output_file, index=False)

    logger.info(f"Results saved to {output_file}")

    # Save calibration data if available
    if calibration_data:
        import pickle

        calibration_file = output_dir / f"county_{args.fips}_calibration.pkl"

        # Organize calibration data
        cal_output = {
            'fips': args.fips,
            'experiment_name': config.get('experiment', {}).get('name'),
            'quantiles': config['calibration']['quantiles'],
            'folds': calibration_data
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(cal_output, f)

        logger.info(f"Calibration data saved to {calibration_file}")

    # MODIFIED: Save prediction data if available
    if predictions_data:
        pred_format = config.get('predictions', {}).get('predictions_format', 'parquet')

        if pred_format == 'parquet':
            predictions_file = output_dir / f"county_{args.fips}_predictions.parquet"

            # Convert list of dicts to DataFrame for parquet
            pred_df = pd.DataFrame(predictions_data)
            pred_df.to_parquet(predictions_file, index=False)

            logger.info(f"Predictions saved to {predictions_file} ({len(predictions_data)} records)")

        elif pred_format == 'pickle':
            import pickle
            predictions_file = output_dir / f"county_{args.fips}_predictions.pkl"

            # Organize prediction data (similar to calibration format)
            pred_output = {
                'fips': args.fips,
                'experiment_name': config.get('experiment', {}).get('name'),
                'predictions': predictions_data
            }

            with open(predictions_file, 'wb') as f:
                pickle.dump(pred_output, f)

            logger.info(f"Predictions saved to {predictions_file} ({len(predictions_data)} records)")
        else:
            logger.warning(f"Unknown predictions format: {pred_format}. Skipping prediction save.")


if __name__ == '__main__':
    main()
