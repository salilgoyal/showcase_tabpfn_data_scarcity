"""
Cross-county generalization experiment: Train on specified train set, test on held-out counties.

This experiment supports two modes:
1. New mode (recommended): Use test_set_config and train_set_config files
   - Fixed test counties with temporal split
   - Configurable training strategies (test history, external counties, etc.)

2. Legacy mode: Train on N-1 counties, test on 1 county (leave-one-out)
   - For backward compatibility with old configs

The experiment uses pre-cleaned pooled data and applies Phase 2 preprocessing
(winsorization, normalization, imputation) per train/test split to avoid leakage.
"""

import pandas as pd
import numpy as np
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseExperimentRunner, ExperimentMetadata
from src.data import CleanedDataLoader
from src.data.split_strategies import (
    create_test_set,
    create_train_set,
    get_train_test_data,
    load_test_set_config,
    load_train_set_config,
    load_test_set_result,
    load_train_set_result,
    save_test_set_result,
    save_train_set_result,
    TestSetResult,
    TrainSetResult
)
from src.data.preprocessing_utils import apply_phase2_preprocessing
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


class CrossCountyExperiment(BaseExperimentRunner):
    """
    Experiment that tests cross-county generalization.

    Supports two modes:
    1. New mode: Uses test_set_config and train_set_config for flexible test/train splits
    2. Legacy mode: Leave-one-out cross-validation across counties
    """

    def __init__(self, config: Dict):
        """
        Initialize cross-county experiment.

        Args:
            config: Configuration dictionary with:
                - data: Dataset paths and settings (cleaned_data_path)
                - test_set_config: Path to test set YAML (optional, new mode)
                - train_set_config: Path to train set YAML (optional, new mode)
                - preprocessing.phase2_steps: Phase 2 preprocessing config
                - experiment: Experiment metadata and settings
                - models: Model configurations
        """
        super().__init__(config)

        self.metadata = ExperimentMetadata(config)

        # Detect mode: new (splits) or legacy (leave-one-out)
        # New mode can use either:
        #   1. Pre-generated splits (recommended): splits.test_set_dir + splits.train_set_dir
        #   2. Generate-on-the-fly: test_set_config + train_set_config
        self.use_new_mode = (
            'splits' in config or
            ('test_set_config' in config and 'train_set_config' in config)
        )

        if self.use_new_mode:
            logger.info("Using NEW mode: test/train splits")
            self._init_new_mode(config)
        else:
            logger.info("Using LEGACY mode: leave-one-out cross-validation")
            self._init_legacy_mode(config)

    def _init_new_mode(self, config: Dict):
        """Initialize for new mode with test/train splits."""
        # Determine if using pre-generated splits or generating on-the-fly
        if 'splits' in config:
            # Mode 1: Load pre-generated splits (recommended)
            self.use_pregenerated = True
            self.test_split_dir = config['splits']['test_set_dir']
            self.train_split_dir = config['splits']['train_set_dir']
            logger.info(f"Using pre-generated splits:")
            logger.info(f"  Test: {self.test_split_dir}")
            logger.info(f"  Train: {self.train_split_dir}")
        else:
            # Mode 2: Generate splits on-the-fly (for development/testing)
            self.use_pregenerated = False
            test_config_path = config['test_set_config']
            train_config_path = config['train_set_config']

            self.test_config = load_test_set_config(test_config_path)
            self.train_config = load_train_set_config(train_config_path)

            logger.info(f"Generating splits on-the-fly:")
            logger.info(f"  Test config: {self.test_config.get('version', 'unknown')}")
            logger.info(f"  Train config: {self.train_config.get('version', 'unknown')}")

        # Phase 2 preprocessing config
        self.phase2_config = config.get('preprocessing', {}).get('phase2_steps', {})

        # Initialize simple data loader (just for loading, not for splits)
        self.data_loader = CleanedDataLoader(
            cleaned_data_path=config['data']['cleaned_data_path'],
            target_column=config['data']['target_column'],
            phase2_config={}  # Don't apply Phase 2 in loader
        )

        # Override log_transformed based on metadata
        self.log_transformed = self.data_loader.is_target_log_transformed()
        logger.info(f"Target log-transformed: {self.log_transformed}")

        # Store target column name
        self.target_column = config['data']['target_column']

        # Will be populated when we load/create splits
        self.test_result: Optional[TestSetResult] = None
        self.train_result: Optional[TrainSetResult] = None

    def _init_legacy_mode(self, config: Dict):
        """Initialize for legacy leave-one-out mode."""
        self.iterations = config.get('iterations', 10)

        # Phase 2 preprocessing config
        self.phase2_config = config.get('preprocessing', {}).get('phase2_steps', {})

        self.data_loader = CleanedDataLoader(
            cleaned_data_path=config['data']['cleaned_data_path'],
            target_column=config['data']['target_column'],
            phase2_config=self.phase2_config
        )

        # Get county list from config or from data
        self.county_fips_list = config.get('county_fips_list', None)
        if self.county_fips_list is None:
            self.county_fips_list = self.data_loader.get_county_fips_list()
            logger.info(f"Using all {len(self.county_fips_list)} counties from cleaned data")

        # Override log_transformed based on metadata
        self.log_transformed = self.data_loader.is_target_log_transformed()
        logger.info(f"Target log-transformed: {self.log_transformed}")

        # Sampling configuration (optional)
        self.sample_train = config.get('sampling', {}).get('max_train_samples', None)
        self.sample_test = config.get('sampling', {}).get('max_test_samples', None)

        self.target_column = config['data']['target_column']

    def _check_log_transform(self) -> bool:
        """Override to get from data metadata instead of config."""
        return False

    # ==========================================================================
    # NEW MODE: Test/Train Set Configs
    # ==========================================================================

    def run_experiment_new_mode(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Run experiment using test_set_config and train_set_config.

        Creates test/train splits based on configs, then evaluates models
        on each test county.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        logger.info("=" * 80)
        logger.info(f"CROSS-COUNTY EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("=" * 80)

        # Load or create test/train splits
        if self.use_pregenerated:
            # Load pre-generated splits (just indices, no data yet)
            logger.info("Loading pre-generated test set...")
            self.test_result = load_test_set_result(self.test_split_dir)

            logger.info("Loading pre-generated train set...")
            self.train_result = load_train_set_result(self.train_split_dir)

            # Check if debug mode (for smoke tests)
            debug_max_rows = None
            if 'debug' in self.config and 'sample_size' in self.config['debug']:
                debug_max_rows = self.config['debug']['sample_size']

            # Load data
            if debug_max_rows is not None:
                # Debug mode: just read first N rows, ignore indices
                df = self.data_loader.load_data_by_indices(np.array([]), max_rows=debug_max_rows)
                # Create dummy indices matching the loaded rows
                self.test_result.test_indices = np.arange(min(debug_max_rows // 2, len(df)))
                self.test_result.train_pool_indices = np.arange(min(debug_max_rows // 2, len(df)))
                self.train_result.train_indices = np.arange(min(debug_max_rows, len(df)))
            else:
                # Normal mode: load specific rows by index
                logger.info("Loading data for pre-generated splits (memory-efficient)...")
                all_indices = np.concatenate([
                    self.test_result.test_indices,
                    self.test_result.train_pool_indices,
                    self.train_result.train_indices
                ])
                # Sort to get the unique indices and create a mapping
                unique_indices = np.unique(all_indices)
                df = self.data_loader.load_data_by_indices(unique_indices)
                logger.info(f"  Loaded {len(df):,} rows (only needed rows, not full dataset)")

                # Create index mapping: old_index -> new_index in the subset DataFrame
                index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

                # Save original test indices before remapping (for predictions files)
                self.original_test_indices = self.test_result.test_indices.copy()

                # Remap indices in test_result and train_result to match subset DataFrame
                self.test_result.test_indices = np.array([index_map[idx] for idx in self.test_result.test_indices])
                self.test_result.train_pool_indices = np.array([index_map[idx] for idx in self.test_result.train_pool_indices])
                self.train_result.train_indices = np.array([index_map[idx] for idx in self.train_result.train_indices])

            # Reset DataFrame index to match
            df = df.reset_index(drop=True)
        else:
            # Generate splits on-the-fly (requires full dataset)
            logger.info("Loading cleaned data...")
            df = self.data_loader.load_data()

            random_seed = self.config['experiment'].get('random_seed', 42)

            logger.info("Creating test set...")
            self.test_result = create_test_set(
                df=df,
                config=self.test_config,
                fips_column="fips",
                date_column="sale_date",
                random_seed=random_seed
            )

            logger.info("Creating train set...")
            self.train_result = create_train_set(
                df=df,
                config=self.train_config,
                test_result=self.test_result,
                fips_column="fips",
                random_seed=random_seed
            )

        # Filter to specific size buckets if configured
        filter_buckets = self.config.get('filter_buckets', None)
        if filter_buckets:
            logger.info(f"Filtering to size buckets: {filter_buckets}")
            target_fips = set()
            for bucket in filter_buckets:
                bucket_counties = self.test_result.size_buckets.get(bucket, [])
                target_fips.update(bucket_counties)
                logger.info(f"  Bucket '{bucket}': {len(bucket_counties)} counties")

            if not target_fips:
                raise ValueError(f"No counties found for buckets: {filter_buckets}")

            # Filter test indices
            test_df_tmp = df.iloc[self.test_result.test_indices]
            test_mask = test_df_tmp['fips'].isin(target_fips).values
            self.test_result.test_indices = self.test_result.test_indices[test_mask]
            # Also filter original_test_indices (pre-remap, same order) used for predictions file
            if hasattr(self, 'original_test_indices'):
                self.original_test_indices = self.original_test_indices[test_mask]

            # Filter train indices
            train_df_tmp = df.iloc[self.train_result.train_indices]
            train_mask = train_df_tmp['fips'].isin(target_fips).values
            self.train_result.train_indices = self.train_result.train_indices[train_mask]

            # Filter test_counties list and county_info
            self.test_result.test_counties = [f for f in self.test_result.test_counties if f in target_fips]
            self.test_result.county_info = {
                k: v for k, v in self.test_result.county_info.items() if k in target_fips
            }

            # Update county_distribution
            self.train_result.county_distribution = {
                k: v for k, v in self.train_result.county_distribution.items() if k in target_fips
            }

            logger.info(f"After filtering: {len(self.test_result.test_counties)} counties, "
                        f"{len(self.test_result.test_indices)} test samples, "
                        f"{len(self.train_result.train_indices)} train samples")

        # Log split info
        logger.info(f"Test counties: {len(self.test_result.test_counties)}")
        logger.info(f"Test samples: {len(self.test_result.test_indices)}")
        logger.info(f"Train samples: {len(self.train_result.train_indices)}")
        logger.info(f"Train source breakdown: {self.train_result.source_breakdown}")
        logger.info("=" * 80)

        # Save fips for train/test before get_train_test_data drops them
        onehot_fips = self.config.get('features', {}).get('onehot_fips', False)
        if onehot_fips:
            train_fips_values = df.iloc[self.train_result.train_indices]['fips'].values
            test_fips_values = df.iloc[self.test_result.test_indices]['fips'].values

        # Get train/test data
        # Use default exclude_columns (None) to get the full list from get_train_test_data
        # This ensures we exclude CALCULATED_TOTAL_VALUE and other baseline features
        X_train, y_train, X_test, y_test = get_train_test_data(
            df=df,
            test_result=self.test_result,
            train_result=self.train_result,
            target_column=self.target_column,
            exclude_columns=None  # Use default list (excludes CALCULATED_TOTAL_VALUE, etc.)
        )

        # Extract fips and size_bucket for each test sample (for predictions files)
        test_df = df.iloc[self.test_result.test_indices]
        self.test_fips = test_df['fips'].values
        self.test_size_buckets = np.array([
            self.test_result.county_info.get(fips, {}).get('size_bucket', 'unknown')
            for fips in self.test_fips
        ])
        logger.info(f"Extracted metadata for {len(self.test_fips)} test samples")

        # Apply Phase 2 preprocessing
        if self.phase2_config:
            logger.info("Applying Phase 2 preprocessing...")
            X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=self.phase2_config
            )

        # One-hot encode FIPS after Phase 2 (so binary columns skip winsorization/normalization)
        if onehot_fips:
            logger.info("One-hot encoding FIPS column...")
            fips_train_df = pd.DataFrame({'fips': train_fips_values}, index=X_train.index)
            fips_test_df = pd.DataFrame({'fips': test_fips_values}, index=X_test.index)

            fips_train_onehot = pd.get_dummies(fips_train_df['fips'], prefix='fips').astype(float)
            fips_test_onehot = pd.get_dummies(fips_test_df['fips'], prefix='fips').astype(float)

            # Align columns: test may have FIPS not in train and vice versa
            fips_train_onehot, fips_test_onehot = fips_train_onehot.align(
                fips_test_onehot, join='outer', axis=1, fill_value=0
            )

            X_train = pd.concat([X_train, fips_train_onehot], axis=1)
            X_test = pd.concat([X_test, fips_test_onehot], axis=1)
            logger.info(f"  Added {fips_train_onehot.shape[1]} FIPS indicator columns")

        logger.info(f"After Phase 2: {X_train.shape[1]} features")

        # Train and evaluate models
        enabled_models = self.get_enabled_models()
        # Always save predictions for per-county evaluation
        save_predictions = True

        all_results = []
        all_calibration_data = []
        all_predictions_data = []

        start_time = time.time()

        for model_name in enabled_models:
            logger.info(f"\nTraining {model_name}...")

            result, cal_data, pred_data = self.train_and_predict(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                save_predictions=save_predictions
            )

            # Add experiment metadata
            result.update({
                'test_set_version': getattr(self, 'test_config', {}).get('version', 'unknown'),
                'train_set_version': getattr(self, 'train_config', {}).get('version', 'unknown'),
                'n_test_counties': len(self.test_result.test_counties),
                'n_train_source_test_history': self.train_result.source_breakdown.get('test_counties_historical', 0),
                'n_train_source_external': self.train_result.source_breakdown.get('external_counties', 0),
                'status': 'success'
            })
            result = self.metadata.add_to_result(result)
            all_results.append(result)

            if cal_data is not None:
                all_calibration_data.append(cal_data)
            if pred_data is not None:
                all_predictions_data.append(pred_data)

                # Save predictions to disk for per-county evaluation
                results_dir = Path(self.config['output']['results_dir'])
                results_dir.mkdir(parents=True, exist_ok=True)
                pred_file = results_dir / f'predictions_{model_name}.parquet'
                # Use original dataset indices (not remapped subset indices)
                # so test_index is consistent with baseline_predictions.parquet
                test_indices = getattr(self, 'original_test_indices', pred_data['test_indices'])
                pred_df = pd.DataFrame({
                    'test_index': test_indices,
                    'fips': self.test_fips,
                    'size_bucket': self.test_size_buckets,
                    'y_true': pred_data['y_true'],
                    'y_pred': pred_data['y_pred']
                })
                pred_df.to_parquet(pred_file, index=False)
                logger.info(f"  Saved predictions to {pred_file}")

        # Now evaluate per test county for granular results
        logger.info("\nEvaluating per test county...")
        per_county_results = self._evaluate_per_county(
            df=df,
            X_test=X_test,
            y_test=y_test,
            enabled_models=enabled_models
        )
        all_results.extend(per_county_results)

        # Create results DataFrame
        df_results = pd.DataFrame(all_results)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Overall results: {len(enabled_models)}")
        logger.info(f"Per-county results: {len(per_county_results)}")

        return (
            df_results,
            all_calibration_data if all_calibration_data else None,
            all_predictions_data if all_predictions_data else None
        )

    def _evaluate_per_county(
        self,
        df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        enabled_models: List[str]
    ) -> List[Dict]:
        """
        Evaluate trained models per test county for granular results.

        Loads predictions from disk and computes metrics for each county.
        """
        from src.evaluation.metrics import compute_metrics

        results = []
        results_dir = Path(self.config['output']['results_dir'])

        # Load predictions for each model
        predictions = {}
        for model_name in enabled_models:
            pred_file = results_dir / f'predictions_{model_name}.parquet'
            if pred_file.exists():
                pred_df = pd.read_parquet(pred_file)
                # Map test_index back to position in y_test
                # y_test is indexed 0 to len(test_indices)-1
                predictions[model_name] = pred_df['y_pred'].values
                logger.debug(f"  Loaded {len(pred_df)} predictions for {model_name}")
            else:
                logger.warning(f"  Predictions file not found for {model_name}: {pred_file}")

        # Get test county mapping
        test_df = df.iloc[self.test_result.test_indices]
        test_fips = test_df['fips'].values

        for fips in self.test_result.test_counties:
            county_mask = test_fips == fips
            county_info = self.test_result.county_info.get(fips, {})

            if not np.any(county_mask):
                continue

            # Get indices for this county in X_test/y_test
            county_indices = np.where(county_mask)[0]
            y_test_county = y_test.iloc[county_indices]

            for model_name in enabled_models:
                # Get number of samples from this county used in training
                county_train_used = self.train_result.county_distribution.get(fips, 0)

                if model_name not in predictions:
                    # No predictions available - create placeholder
                    result = {
                        'model': model_name,
                        'result_type': 'per_county',
                        'test_fips': fips,
                        'size_bucket': county_info.get('size_bucket', 'unknown'),
                        'county_test_size': len(county_indices),
                        'county_train_pool_size': county_info.get('train_pool_rows', 0),
                        'county_train_used': county_train_used,
                        'test_set_version': getattr(self, 'test_config', {}).get('version', 'unknown'),
                        'train_set_version': getattr(self, 'train_config', {}).get('version', 'unknown'),
                    }
                else:
                    # Get predictions for this county
                    y_pred_county = predictions[model_name][county_indices]

                    # Compute metrics
                    metrics = compute_metrics(
                        y_true=y_test_county.values,
                        y_pred=y_pred_county,
                        log_transformed=self.log_transformed
                    )

                    result = {
                        'model': model_name,
                        'result_type': 'per_county',
                        'test_fips': fips,
                        'size_bucket': county_info.get('size_bucket', 'unknown'),
                        'county_test_size': len(county_indices),
                        'county_train_pool_size': county_info.get('train_pool_rows', 0),
                        'county_train_used': county_train_used,
                        'test_set_version': getattr(self, 'test_config', {}).get('version', 'unknown'),
                        'train_set_version': getattr(self, 'train_config', {}).get('version', 'unknown'),
                        'r2': metrics['r2'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape'],
                        'mse': metrics['mse'],
                    }

                result = self.metadata.add_to_result(result)
                results.append(result)

        return results

    # ==========================================================================
    # LEGACY MODE: Leave-One-Out
    # ==========================================================================

    def run_single_iteration_legacy(
        self,
        target_fips: int,
        iteration: int,
        df: pd.DataFrame
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run one iteration for a target county (legacy mode).

        Args:
            target_fips: Target county FIPS code (held out for testing)
            iteration: Iteration number
            df: Full cleaned DataFrame

        Returns:
            Tuple of (results, calibration_data, predictions_data)
        """
        logger.info(f"  Target county {target_fips}, iteration {iteration}")

        try:
            # Define train/test counties
            train_fips = [f for f in self.county_fips_list if f != target_fips]
            test_fips = [target_fips]

            # Get train/test data with Phase 2 preprocessing
            random_state = self.config.get('experiment', {}).get('random_seed', 42) + iteration
            X_train, y_train, X_test, y_test = self.data_loader.prepare_train_test_split(
                train_fips=train_fips,
                test_fips=test_fips,
                apply_phase2=True,
                sample_train=self.sample_train,
                sample_test=self.sample_test,
                random_state=random_state
            )

            # Get enabled models
            enabled_models = self.get_enabled_models()
            save_predictions = self.config.get('predictions', {}).get('save_predictions', False)

            # Train and evaluate each model
            results = []
            calibration_data = []
            predictions_data = []

            for model_name in enabled_models:
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
                    'target_fips': target_fips,
                    'iteration': iteration,
                    'n_train_counties': len(train_fips),
                    'status': 'success'
                })
                result = self.metadata.add_to_result(result)
                results.append(result)

                if cal_data is not None:
                    cal_data['target_fips'] = target_fips
                    cal_data['iteration'] = iteration
                    calibration_data.append(cal_data)

                if pred_data is not None:
                    pred_data['target_fips'] = target_fips
                    pred_data['iteration'] = iteration
                    predictions_data.append(pred_data)

            return results, calibration_data, predictions_data

        except Exception as e:
            logger.error(f"Failed for target {target_fips}, iteration {iteration}: {e}", exc_info=True)

            # Return failed results for all models
            enabled_models = self.get_enabled_models()
            results = []
            for model_name in enabled_models:
                result = self.metadata.add_to_result({
                    'target_fips': target_fips,
                    'iteration': iteration,
                    'n_train_counties': len(self.county_fips_list) - 1,
                    'model': model_name,
                    'train_size': 0,
                    'test_size': 0,
                    'n_features': 0,
                    'fit_time': 0,
                    'pred_time': 0,
                    'status': f'failed: {str(e)}',
                    'r2': np.nan,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mse': np.nan
                })
                results.append(result)

            return results, [], []

    def run_experiment_legacy(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Run the full cross-county experiment (legacy leave-one-out mode).

        For each county as target:
            For each iteration:
                Train on all other counties, test on target county

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        logger.info("=" * 80)
        logger.info(f"CROSS-COUNTY EXPERIMENT (LEGACY): {self.config['experiment']['name']}")
        logger.info("=" * 80)

        # Load data once (will be cached)
        logger.info("Loading cleaned data...")
        df = self.data_loader.load_data()

        enabled_models = self.get_enabled_models()
        n_counties = len(self.county_fips_list)
        total_experiments = n_counties * self.iterations * len(enabled_models)

        logger.info(f"Counties: {n_counties}")
        logger.info(f"Iterations per county: {self.iterations}")
        logger.info(f"Models: {enabled_models}")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("=" * 80)

        # Run experiments
        all_results = []
        all_calibration_data = []
        all_predictions_data = []

        experiment_num = 0
        start_time = time.time()

        for target_fips in sorted(self.county_fips_list):
            logger.info(f"\nTarget county: {target_fips}")

            for iteration in range(self.iterations):
                experiment_num += len(enabled_models)
                logger.info(f"  Experiment {experiment_num}/{total_experiments}")

                results, cal_data, pred_data = self.run_single_iteration_legacy(
                    target_fips=target_fips,
                    iteration=iteration,
                    df=df
                )

                all_results.extend(results)
                all_calibration_data.extend(cal_data)
                all_predictions_data.extend(pred_data)

        # Create results DataFrame
        df_results = pd.DataFrame(all_results)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Total experiments: {len(all_results)}")
        logger.info(f"Successful: {sum(1 for r in all_results if r['status'] == 'success')}")

        return (
            df_results,
            all_calibration_data if all_calibration_data else None,
            all_predictions_data if all_predictions_data else None
        )

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    def run_experiment(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Run the cross-county experiment.

        Automatically selects between new mode (test/train configs) and
        legacy mode (leave-one-out) based on config contents.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        if self.use_new_mode:
            return self.run_experiment_new_mode()
        else:
            return self.run_experiment_legacy()
