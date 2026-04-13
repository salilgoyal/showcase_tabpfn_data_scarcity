"""
Data scaling experiment: Vary training data size with fixed test set.

This experiment type is useful for:
1. Understanding data efficiency (learning curves)
2. Comparing models across different data regimes
3. Cook County-style subsampling experiments

The experiment:
- Loads a dataset and splits into train pool and test set
- For each train_size and random seed:
  - Samples train_size examples from the train pool
  - Trains model and evaluates on fixed test set
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runners.base_runner import BaseExperimentRunner, ExperimentMetadata
from data.evelyn_preprocessing import load_and_prepare_data

logger = logging.getLogger(__name__)


class DataScalingExperiment(BaseExperimentRunner):
    """
    Experiment that varies training data size with a fixed test set.

    This replaces the old cook_county_runner.py with a more general implementation
    that can work with any county or dataset.
    """

    def __init__(self, config: Dict):
        """
        Initialize data scaling experiment.

        Args:
            config: Configuration dictionary with:
                - data: Dataset paths and settings
                - train_sizes: List of training sizes to test
                - seeds: List of random seeds for sampling
                - models: Model configurations
                - preprocessing: Preprocessing settings
        """
        super().__init__(config)
        self.train_sizes = config.get('train_sizes', [100, 200, 500, 1000])
        self.seeds = config.get('seeds', list(range(10)))
        self.metadata = ExperimentMetadata(config)

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load dataset and apply preprocessing.

        Returns:
            X_train_pool, y_train_pool, X_test, y_test
        """
        # Get data path (support both cook county and individual counties)
        data_config = self.config['data']

        if 'cook_county_csv' in data_config:
            # Cook County mode
            data_path = data_config['cook_county_csv']
            # Convert relative to absolute if needed
            if not os.path.isabs(data_path):
                project_root = Path(__file__).parent.parent.parent
                data_path = str(project_root / data_path)

            logger.info(f"Loading Cook County data from: {data_path}")

        elif 'county_csv_path' in data_config:
            # Single county mode
            data_path = data_config['county_csv_path']
            logger.info(f"Loading county data from: {data_path}")

        else:
            raise ValueError("Config must specify either 'cook_county_csv' or 'county_csv_path'")

        # Use preprocessing pipeline
        feature_config = self.config.get('preprocessing', {}).get('features', {})
        step_config = self.config.get('preprocessing', {}).get('steps', {})

        X_train_pool, y_train_pool, X_test, y_test, _ = load_and_prepare_data(
            data_path=data_path,
            feature_config=feature_config,
            step_config=step_config,
            cbg_column=data_config.get('cbg_column', 'block_group_id')
        )

        logger.info(f"Features: {X_train_pool.shape[1]}")
        logger.info(f"Training pool: {len(X_train_pool)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train_pool, y_train_pool, X_test, y_test

    def sample_train_set(
        self,
        X_train_pool: pd.DataFrame,
        y_train_pool: pd.Series,
        train_size: int,
        seed: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sample training set of specified size.

        Args:
            X_train_pool: Full training pool features
            y_train_pool: Full training pool targets
            train_size: Number of samples to draw
            seed: Random seed

        Returns:
            X_train, y_train
        """
        np.random.seed(seed)

        if train_size >= len(X_train_pool):
            logger.warning(
                f"Requested train_size {train_size} >= pool size {len(X_train_pool)}, "
                f"using full pool"
            )
            return X_train_pool, y_train_pool

        # Sample indices
        indices = np.random.choice(len(X_train_pool), size=train_size, replace=False)

        return X_train_pool.iloc[indices], y_train_pool.iloc[indices]

    def run_single_iteration(
        self,
        model_name: str,
        train_size: int,
        seed: int,
        X_train_pool: pd.DataFrame,
        y_train_pool: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Run a single experiment iteration.

        Args:
            model_name: Model to use
            train_size: Number of training samples
            seed: Random seed
            X_train_pool: Full training pool
            y_train_pool: Full training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Result dictionary
        """
        logger.info(f"Running {model_name} with train_size={train_size}, seed={seed}")

        try:
            # Sample training data
            X_train, y_train = self.sample_train_set(X_train_pool, y_train_pool, train_size, seed)

            # Check if predictions should be saved
            save_predictions = self.config.get('predictions', {}).get('save_predictions', False)

            # Train and predict
            result, cal_data, pred_data = self.train_and_predict(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                save_predictions=save_predictions
            )

            # Add experiment-specific metadata
            result.update({
                'train_size': train_size,
                'seed': seed,
                'status': 'success'
            })
            result = self.metadata.add_to_result(result)

            logger.info(f"Success: R²={result['r2']:.4f}, MAE={result['mae']:.2f}")

            return result, cal_data, pred_data

        except Exception as e:
            logger.error(f"Failed: {str(e)}", exc_info=True)

            # Return failed result
            result = self.metadata.add_to_result({
                'train_size': train_size,
                'test_size': len(y_test),
                'n_features': 0,
                'seed': seed,
                'model': model_name,
                'fit_time': 0,
                'pred_time': 0,
                'status': f'failed: {str(e)}',
                'r2': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'mse': np.nan
            })

            return result, None, None

    def run_experiment(self) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        Run the full data scaling experiment.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        logger.info("=" * 80)
        logger.info(f"DATA SCALING EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("=" * 80)

        # Load and prepare data
        X_train_pool, y_train_pool, X_test, y_test = self.load_and_prepare_data()

        # Get enabled models
        enabled_models = self.get_enabled_models()

        logger.info(f"Models: {enabled_models}")
        logger.info(f"Train sizes: {self.train_sizes}")
        logger.info(f"Seeds: {self.seeds}")

        total_experiments = len(enabled_models) * len(self.train_sizes) * len(self.seeds)
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("=" * 80)

        # Run experiments
        results = []
        calibration_data = []
        predictions_data = []

        experiment_num = 0
        import time
        start_time = time.time()

        for model_name in enabled_models:
            for train_size in self.train_sizes:
                for seed in self.seeds:
                    experiment_num += 1
                    logger.info(f"\n--- Experiment {experiment_num}/{total_experiments} ---")

                    result, cal_data, pred_data = self.run_single_iteration(
                        model_name, train_size, seed,
                        X_train_pool, y_train_pool, X_test, y_test
                    )

                    results.append(result)

                    if cal_data is not None:
                        cal_data['train_size'] = train_size
                        cal_data['seed'] = seed
                        calibration_data.append(cal_data)

                    if pred_data is not None:
                        pred_data['train_size'] = train_size
                        pred_data['seed'] = seed
                        predictions_data.append(pred_data)

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {sum(r['status'] == 'success' for r in results)}")

        return df_results, calibration_data or None, predictions_data or None
