"""
Base experiment runner with shared logic for all experiment types.

This module provides a foundation for different experiment types (within-county CV,
cross-county, data scaling, in-context pooling, fine-tuning, etc.) by extracting
common functionality:
- Model initialization and cleanup
- Training and prediction pipeline
- Metrics computation
- Result saving (CSV, predictions, calibration)
- Config management

Experiment-specific runners inherit from BaseExperimentRunner and implement
their own data splitting and experiment loop logic.
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from src.models import TabPFNModel, TabICLModel, XGBoostModel, BaselineModel
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


class BaseExperimentRunner(ABC):
    """
    Abstract base class for experiment runners.

    Provides shared functionality for model training, evaluation, and result management.
    Subclasses implement experiment-specific data splitting and loop logic.
    """

    def __init__(self, config: Dict):
        """
        Initialize base runner with configuration.

        Args:
            config: Configuration dictionary containing experiment parameters
        """
        self.config = config
        self.log_transformed = self._check_log_transform()

    def _check_log_transform(self) -> bool:
        """Check if target is log-transformed based on config."""
        return self.config.get('preprocessing', {}).get('steps', {}).get('log_transform_target', False)

    def create_model(self, model_name: str):
        """
        Create and initialize a model.

        Args:
            model_name: Name of model ('tabpfn' or 'xgboost')

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name == 'tabpfn':
            tabpfn_config = self.config.get('tabpfn', {})
            model = TabPFNModel(
                device=tabpfn_config.get('device', 'cuda'),
                version=tabpfn_config.get('version', 'v2'),  # Default to v2 for backward compatibility
                random_state=self.config.get('experiment', {}).get('random_seed', 42)
            )
        elif model_name == 'tabpfn_v2.5':
            tabpfn25_config = self.config.get('tabpfn_v2.5', {})
            model = TabPFNModel(
                device=tabpfn25_config.get('device', 'cuda'),
                version='v2.5',
                random_state=self.config.get('experiment', {}).get('random_seed', 42)
            )
        elif model_name == 'tabicl':
            tabicl_config = self.config.get('tabicl', {})
            model = TabICLModel(
                device=tabicl_config.get('device', None),
                n_estimators=tabicl_config.get('n_estimators', 8),
                random_state=self.config.get('experiment', {}).get('random_seed', 42)
            )
        elif model_name == 'xgboost':
            xgboost_config = self.config.get('xgboost', {})
            model = XGBoostModel(
                n_trials=xgboost_config.get('optuna_trials', 20),
                cv_folds=xgboost_config.get('optuna_cv_folds', 3),
                use_gpu=xgboost_config.get('use_gpu', True),
                random_state=self.config.get('experiment', {}).get('random_seed', 42)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    def train_and_predict(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_predictions: bool = False
    ) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
        """
        Train a model and make predictions.

        Args:
            model_name: Name of model to use
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            save_predictions: Whether to save individual predictions

        Returns:
            Tuple of (metrics_dict, calibration_data, prediction_data)
        """
        logger.debug(f"  Training {model_name}...")

        # Create model
        model = self.create_model(model_name)

        # Fit model
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start

        # Predict
        pred_start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start

        # Compute metrics
        metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)

        # Build result dictionary
        result = {
            'model': model_name,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[1],
            'fit_time': fit_time,
            'pred_time': pred_time,
            **metrics
        }

        # Add hyperparameters if configured
        hyperparams = None
        if self.config.get('logging', {}).get('log_hyperparameters', False):
            hyperparams = model.get_hyperparameters()
            if hyperparams and model_name == 'xgboost':
                for key, value in hyperparams.items():
                    result[f'hyperparam_{key}'] = value
                if hasattr(model, 'get_tune_time'):
                    result['tune_time'] = model.get_tune_time()

        # Calibration predictions (if enabled)
        calibration_data = self._get_calibration_data(
            model, model_name, X_test, y_test, y_pred
        )

        # Individual predictions (if enabled)
        prediction_data = None
        if save_predictions:
            prediction_data = {
                'model': model_name,
                'test_indices': y_test.index.values,
                'y_true': y_test.values,
                'y_pred': y_pred
            }

        # Cleanup
        model.cleanup()

        logger.debug(
            f"  {model_name}: R2={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.2f}, fit_time={fit_time:.2f}s"
        )

        return result, calibration_data, prediction_data

    def evaluate_baseline(
        self,
        baseline_values_train: np.ndarray,
        baseline_values_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        adjustment_ratio: Optional[float] = None,
        save_predictions: bool = False
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Evaluate baseline model using county assessments.

        Args:
            baseline_values_train: Baseline values for training set
            baseline_values_test: Baseline values for test set
            y_train: Training targets
            y_test: Test targets
            adjustment_ratio: Pre-computed adjustment ratio (optional)
            save_predictions: Whether to save individual predictions

        Returns:
            Tuple of (metrics_dict, prediction_data)
        """
        logger.debug("  Evaluating baseline...")

        # Create baseline model
        model = BaselineModel(
            baseline_values_train=baseline_values_train,
            baseline_values_test=baseline_values_test,
            adjustment_ratio=adjustment_ratio,
            log_transformed=self.log_transformed
        )

        # Fit model (computes adjustment ratio if not provided)
        fit_start = time.time()
        # Baseline doesn't use X_train, but we pass dummy DataFrame for interface compatibility
        model.fit(pd.DataFrame(), y_train)
        fit_time = time.time() - fit_start

        # Predict
        pred_start = time.time()
        # Baseline doesn't use X_test, but we pass dummy DataFrame for interface compatibility
        y_pred = model.predict(pd.DataFrame())
        pred_time = time.time() - pred_start

        # Compute metrics
        metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)

        # Build result dictionary
        result = {
            'model': 'baseline',
            'train_size': len(y_train),
            'test_size': len(y_test),
            'n_features': 0,  # Baseline doesn't use features
            'fit_time': fit_time,
            'pred_time': pred_time,
            **metrics
        }

        # Add adjustment ratio as "hyperparameter"
        hyperparams = model.get_hyperparameters()
        if hyperparams:
            result['adjustment_ratio'] = hyperparams['adjustment_ratio']

        # Individual predictions (if enabled)
        prediction_data = None
        if save_predictions:
            prediction_data = {
                'model': 'baseline',
                'test_indices': y_test.index.values,
                'y_true': y_test.values,
                'y_pred': y_pred
            }

        logger.debug(
            f"  baseline: R2={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.2f}, ratio={hyperparams.get('adjustment_ratio', 'N/A'):.4f}"
        )

        return result, prediction_data

    def load_precomputed_baseline_results(self, train_split_dir: str) -> Optional[List[Dict]]:
        """
        Load pre-computed baseline results if available.

        Args:
            train_split_dir: Path to train split directory

        Returns:
            List of result dictionaries (overall + per-county) or None if not found
        """
        baseline_file = Path(train_split_dir) / "baseline_results.csv"
        if not baseline_file.exists():
            return None

        logger.info(f"Loading pre-computed baseline results from {baseline_file}")
        baseline_df = pd.read_csv(baseline_file)

        # Convert DataFrame rows to dictionaries
        results = []
        for _, row in baseline_df.iterrows():
            result = row.to_dict()
            # Convert empty strings to None/NaN
            for key, value in result.items():
                if value == '':
                    result[key] = None
            results.append(result)

        logger.info(f"Loaded {len(results)} pre-computed baseline result rows")
        return results

    def _get_calibration_data(
        self,
        model,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Optional[Dict]:
        """
        Get calibration data if calibration is enabled.

        Args:
            model: Trained model instance
            model_name: Name of the model
            X_test: Test features
            y_test: Test targets
            y_pred: Predictions

        Returns:
            Calibration data dict or None
        """
        calibration_enabled = self.config.get('calibration', {}).get('enabled', False)
        if not calibration_enabled or model_name != 'tabpfn':
            return None

        quantiles = self.config.get('calibration', {}).get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        logger.debug(f"  Predicting quantiles for calibration: {quantiles}")
        quantile_preds = model.predict_quantiles(X_test, quantiles)

        return {
            'model': model_name,
            'y_true': y_test.values,
            'y_pred_mean': y_pred,
            'quantile_predictions': quantile_preds
        }

    def get_enabled_models(self, include_baseline: bool = False) -> List[str]:
        """
        Get list of enabled models from config.

        Args:
            include_baseline: If True, include 'baseline' in the list if enabled

        Returns:
            List of model names that are enabled
        """
        model_configs = self.config['models']
        enabled_models = [m['name'] for m in model_configs if m['enabled']]

        # Filter for calibration experiments if needed
        calibration_enabled = self.config.get('calibration', {}).get('enabled', False)
        if calibration_enabled:
            calibration_models = self.config.get('calibration', {}).get('models', ['tabpfn'])
            enabled_models = [m for m in enabled_models if m in calibration_models]
            logger.info(f"Calibration mode: only running models {enabled_models}")

        # Add baseline if requested and enabled
        if include_baseline and self.is_baseline_enabled():
            enabled_models.append('baseline')

        return enabled_models

    def is_baseline_enabled(self) -> bool:
        """
        Check if baseline comparison is enabled in config.

        Returns:
            True if baseline is enabled, False otherwise
        """
        return self.config.get('baseline', {}).get('enabled', False)

    def save_results(
        self,
        results_df: pd.DataFrame,
        output_dir: Path,
        filename: str
    ):
        """
        Save results DataFrame to CSV.

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
            filename: Output filename
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    def save_calibration_data(
        self,
        calibration_data: List[Dict],
        output_dir: Path,
        filename: str
    ):
        """
        Save calibration data to pickle file.

        Args:
            calibration_data: List of calibration data dictionaries
            output_dir: Output directory
            filename: Output filename
        """
        if not calibration_data:
            return

        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_file = output_dir / filename

        cal_output = {
            'experiment_name': self.config.get('experiment', {}).get('name'),
            'quantiles': self.config.get('calibration', {}).get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9]),
            'data': calibration_data
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(cal_output, f)

        logger.info(f"Calibration data saved to {calibration_file}")

    def save_predictions(
        self,
        predictions_data: List[Dict],
        output_dir: Path,
        filename: str
    ):
        """
        Save individual predictions to parquet or pickle.

        Args:
            predictions_data: List of prediction data dictionaries
            output_dir: Output directory
            filename: Output filename (with extension)
        """
        if not predictions_data:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        pred_format = self.config.get('predictions', {}).get('predictions_format', 'parquet')

        if pred_format == 'parquet':
            predictions_file = output_dir / filename.replace('.pkl', '.parquet')
            pred_df = pd.DataFrame(predictions_data)
            pred_df.to_parquet(predictions_file, index=False)
            logger.info(f"Predictions saved to {predictions_file} ({len(predictions_data)} records)")

        elif pred_format == 'pickle':
            import pickle
            predictions_file = output_dir / filename

            pred_output = {
                'experiment_name': self.config.get('experiment', {}).get('name'),
                'predictions': predictions_data
            }

            with open(predictions_file, 'wb') as f:
                pickle.dump(pred_output, f)

            logger.info(f"Predictions saved to {predictions_file} ({len(predictions_data)} records)")

    @abstractmethod
    def run_experiment(self):
        """
        Run the experiment. Must be implemented by subclasses.

        This method should:
        1. Load/prepare data according to experiment type
        2. Loop over splits/conditions
        3. Call train_and_predict() for each configuration
        4. Collect and return results

        Returns:
            Results DataFrame and any additional outputs (calibration, predictions, etc.)
        """
        pass


class ExperimentMetadata:
    """Helper class for managing experiment metadata."""

    def __init__(self, config: Dict):
        self.config = config

    def get_experiment_info(self) -> Dict:
        """Get common experiment metadata fields."""
        return {
            'experiment_name': self.config.get('experiment', {}).get('name', 'default'),
            'experiment_description': self.config.get('experiment', {}).get('description', ''),
        }

    def add_to_result(self, result: Dict) -> Dict:
        """Add experiment metadata to a result dictionary."""
        result.update(self.get_experiment_info())
        return result
