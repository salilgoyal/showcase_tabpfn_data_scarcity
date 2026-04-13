"""
Fine-tuning experiment: Train on specified train set with fine-tuned TabPFN, test on held-out counties.

This experiment is identical to CrossCountyExperiment except it uses fine-tuned TabPFN
instead of the standard TabPFN model. It:
1. Uses the same test_set_config and train_set_config structure
2. Uses CleanedDataLoader for data loading
3. Applies the same Phase 2 preprocessing
4. Trains XGBoost identically
5. Trains TabPFN using fine-tuning with train/val split

The experiment uses pre-cleaned pooled data and applies Phase 2 preprocessing
(winsorization, normalization, imputation) per train/test split to avoid leakage.
"""

import pandas as pd
import numpy as np
import logging
import time
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
from src.data.preprocessing_utils import apply_phase2_preprocessing, Phase2Preprocessor
from src.evaluation import compute_metrics
from src.models.tabpfn_finetuning import FineTunedTabPFNModel, FinetuningConfig
from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel, FinetuningConfigV2

logger = logging.getLogger(__name__)


class FinetuningExperiment(BaseExperimentRunner):
    """
    Fine-tuning experiment that tests cross-county generalization using fine-tuned TabPFN.

    This is identical to CrossCountyExperiment except:
    - TabPFN model is fine-tuned using train/val split
    - Training history and checkpoints are saved
    """

    def __init__(self, config: Dict):
        """
        Initialize fine-tuning experiment.

        Args:
            config: Configuration dictionary with:
                - data: Dataset paths and settings (cleaned_data_path)
                - test_set_config: Path to test set YAML (optional, new mode)
                - train_set_config: Path to train set YAML (optional, new mode)
                - preprocessing.phase2_steps: Phase 2 preprocessing config
                - experiment: Experiment metadata and settings
                - models: Model configurations
                - finetuning: Fine-tuning configuration (learning_rate, max_epochs, etc.)
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
        logger.info(f"FINETUNING EXPERIMENT: {self.config['experiment']['name']}")
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

        # Log split info
        logger.info(f"Test counties: {len(self.test_result.test_counties)}")
        logger.info(f"Test samples: {len(self.test_result.test_indices)}")
        logger.info(f"Train samples: {len(self.train_result.train_indices)}")
        logger.info(f"Train source breakdown: {self.train_result.source_breakdown}")
        logger.info("=" * 80)

        # Extract baseline values before get_train_test_data() drops CALCULATED_TOTAL_VALUE
        baseline_column = "CALCULATED_TOTAL_VALUE"
        baseline_enabled = self.is_baseline_enabled() and baseline_column in df.columns
        baseline_values_train = None
        baseline_values_test = None

        if baseline_enabled:
            baseline_values_train = df.iloc[self.train_result.train_indices][baseline_column].values
            baseline_values_test = df.iloc[self.test_result.test_indices][baseline_column].values
            logger.info(f"Extracted baseline values: {len(baseline_values_train)} train, {len(baseline_values_test)} test")

        # Get train/test data
        # Use default exclude_columns (None) to get the full list from get_train_test_data
        # This ensures X_train has the same columns as in-context data
        X_train, y_train, X_test, y_test = get_train_test_data(
            df=df,
            test_result=self.test_result,
            train_result=self.train_result,
            target_column=self.target_column,
            exclude_columns=None  # Use default list to match in-context preprocessing
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
            # Create and save the preprocessor so we can use it for in-context data
            self.phase2_preprocessor = Phase2Preprocessor(self.phase2_config)
            self.phase2_preprocessor.fit(X_train, y_train)

            X_train = self.phase2_preprocessor.transform(X_train)
            y_train = self.phase2_preprocessor.transform_target(y_train)
            X_test = self.phase2_preprocessor.transform(X_test)
            y_test = self.phase2_preprocessor.transform_target(y_test)
        else:
            self.phase2_preprocessor = None

        logger.info(f"After Phase 2: {X_train.shape[1]} features")

        # Train and evaluate models
        enabled_models = self.get_enabled_models()
        # Always save predictions for per-county evaluation
        save_predictions = True  # Force to True for per-county evaluation

        all_results = []
        all_calibration_data = []
        all_predictions_data = []

        # Evaluate baseline model (uses winsorized y_train/y_test)
        if baseline_enabled:
            logger.info("\nEvaluating baseline model...")
            baseline_result, baseline_pred_data = self.evaluate_baseline(
                baseline_values_train=baseline_values_train,
                baseline_values_test=baseline_values_test,
                y_train=y_train,
                y_test=y_test,
                save_predictions=True
            )

            # Add experiment metadata (same pattern as model loop)
            baseline_result.update({
                'test_set_version': getattr(self, 'test_config', {}).get('version', 'unknown'),
                'train_set_version': getattr(self, 'train_config', {}).get('version', 'unknown'),
                'n_test_counties': len(self.test_result.test_counties),
                'n_train_source_test_history': self.train_result.source_breakdown.get('test_counties_historical', 0),
                'n_train_source_external': self.train_result.source_breakdown.get('external_counties', 0),
                'status': 'success'
            })
            baseline_result = self.metadata.add_to_result(baseline_result)
            all_results.append(baseline_result)

            # Save predictions to disk (same format as other models)
            if baseline_pred_data is not None:
                results_dir = Path(self.config['output']['results_dir'])
                results_dir.mkdir(parents=True, exist_ok=True)
                pred_file = results_dir / 'predictions_baseline.parquet'
                pred_df = pd.DataFrame({
                    'test_index': baseline_pred_data['test_indices'],
                    'fips': self.test_fips,
                    'size_bucket': self.test_size_buckets,
                    'y_true': baseline_pred_data['y_true'],
                    'y_pred': baseline_pred_data['y_pred']
                })
                pred_df.to_parquet(pred_file, index=False)
                logger.info(f"  Saved baseline predictions to {pred_file}")

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

                # Save predictions per model for per-county evaluation
                results_dir = Path(self.config['output']['results_dir'])
                results_dir.mkdir(parents=True, exist_ok=True)
                pred_file = results_dir / f'predictions_{model_name}.parquet'
                pred_df = pd.DataFrame({
                    'test_index': pred_data['test_indices'],
                    'fips': self.test_fips,
                    'size_bucket': self.test_size_buckets,
                    'y_true': pred_data['y_true'],
                    'y_pred': pred_data['y_pred']
                })
                pred_df.to_parquet(pred_file, index=False)
                logger.info(f"  Saved predictions to {pred_file}")

        # Now evaluate per test county for granular results
        logger.info("\nEvaluating per test county...")
        # Include baseline in per-county evaluation
        per_county_models = list(enabled_models)
        if baseline_enabled:
            per_county_models.append('baseline')

        per_county_results = self._evaluate_per_county(
            df=df,
            X_test=X_test,
            y_test=y_test,
            enabled_models=per_county_models
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

    def _select_in_context_samples(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: Dict
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load in-context samples from a pre-generated dataset.

        Args:
            X_train: Training features (for fallback if dataset_dir not specified)
            y_train: Training targets (for fallback if dataset_dir not specified)
            config: In-context configuration dict

        Returns:
            Tuple of (X_context, y_context)
        """
        from src.data import load_train_set_result

        dataset_dir = config.get('dataset_dir', None)
        n_samples = config.get('n_samples', None)

        if dataset_dir is None:
            # Fallback: use training data (first 100 samples)
            logger.warning("No in_context.dataset_dir specified, using first 100 training samples")
            n_samples = 100 if n_samples is None else min(n_samples, len(X_train))
            return X_train.iloc[:n_samples], y_train.iloc[:n_samples]

        # Load the pre-generated in-context dataset
        logger.info(f"Loading in-context dataset from: {dataset_dir}")
        try:
            from pathlib import Path
            context_train_result = load_train_set_result(dataset_dir)

            # Get context indices from the loaded train result
            context_indices = context_train_result.train_indices

            # Apply n_samples limit if specified
            if n_samples is not None:
                context_indices = context_indices[:n_samples]

            logger.info(f"  Loaded {len(context_indices)} in-context samples from pre-generated dataset")

            # Get the full dataframe to extract context samples
            # Note: self.data_loader.df should be available if we're using NEW mode
            if hasattr(self.data_loader, 'df') and self.data_loader.df is not None:
                df = self.data_loader.df
            else:
                # Load data if not already loaded
                logger.info("  Loading full dataset to extract in-context samples...")
                df = self.data_loader.load_data()

            # Extract context data
            context_df = df.iloc[context_indices]

            # Apply the same preprocessing (Phase 2) that was applied to X_train
            # Get target and features
            target_column = self.config['data']['target_column']
            y_context = context_df[target_column]

            # Drop target and excluded columns that exist (same as training data)
            # This list MUST match the default in get_train_test_data() in split_strategies.py
            exclude_columns = [
                "fips", "CLIP", "sale_date",
                "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
                "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
                "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
                "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
                "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
                "CALCULATED_TOTAL_VALUE",  # Baseline value - excluded from training features
            ]
            # Only drop columns that actually exist in the dataframe
            columns_to_drop = [target_column] + [c for c in exclude_columns if c in context_df.columns]
            X_context = context_df.drop(columns=columns_to_drop)

            # Drop object columns (same as done in split_strategies.py)
            object_cols = X_context.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                logger.info(f"  Dropping {len(object_cols)} object columns from in-context data: {object_cols}")
                X_context = X_context.drop(columns=object_cols)

            logger.info(f"  X_context shape after dropping columns: {X_context.shape}")

            # Apply Phase 2 preprocessing using the fitted preprocessor
            if hasattr(self, 'phase2_preprocessor') and self.phase2_preprocessor is not None:
                logger.info(f"  Applying Phase 2 preprocessing to in-context data...")

                # X_context should have the same columns as X_train (before Phase 2) because
                # we used the same exclude_columns list. Transform directly.
                X_context = self.phase2_preprocessor.transform(X_context)
                logger.info(f"  X_context shape after Phase 2: {X_context.shape}")
            else:
                logger.warning("  Phase 2 preprocessor not available - cannot properly align columns")
                raise ValueError("Phase 2 preprocessor required but not available")

            return X_context, y_context

        except Exception as e:
            logger.error(f"Failed to load in-context dataset from {dataset_dir}: {e}")
            logger.warning("Falling back to first 100 training samples")
            n_samples = 100 if n_samples is None else min(n_samples, len(X_train))
            return X_train.iloc[:n_samples], y_train.iloc[:n_samples]

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
        Run the full fine-tuning experiment (legacy leave-one-out mode).

        For each county as target:
            For each iteration:
                Train on all other counties, test on target county

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        logger.info("=" * 80)
        logger.info(f"FINETUNING EXPERIMENT (LEGACY): {self.config['experiment']['name']}")
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
    # MODEL CREATION WITH FINE-TUNING SUPPORT
    # ==========================================================================

    def create_model(self, model_name: str):
        """
        Create and initialize a model.

        Override from base class to support fine-tuned TabPFN.

        Args:
            model_name: Name of model ('tabpfn' or 'xgboost')

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name == 'tabpfn':
            implementation = self.config.get('finetuning', {}).get('implementation', 'v1')
            if implementation == 'v2':
                ft_config = self._create_finetuning_config_v2()
                model = DirectFineTunedTabPFNModel(config=ft_config)
            else:
                ft_config = self._create_finetuning_config()
                model = FineTunedTabPFNModel(
                    config=ft_config,
                    device='cuda',
                    random_state=self.config.get('experiment', {}).get('random_seed', 42)
                )
        elif model_name == 'xgboost':
            from src.models import XGBoostModel
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

    def _create_finetuning_config(self) -> FinetuningConfig:
        """Create FinetuningConfig from experiment config."""
        ft_config = self.config.get('finetuning', {})

        # Create checkpoint directory if specified
        checkpoint_dir = None
        if ft_config.get('save_checkpoints', True):
            output_dir = Path(self.config.get('output', {}).get('results_dir', './results'))
            checkpoint_dir = str(output_dir / 'checkpoints')

        return FinetuningConfig(
            learning_rate=float(ft_config.get('learning_rate', 1e-5)),
            learning_rate_schedule=ft_config.get('learning_rate_schedule', 'constant'),
            warmup_epochs=int(ft_config.get('warmup_epochs', 0)),
            max_epochs=int(ft_config.get('max_epochs', 30)),
            batch_size=int(ft_config.get('batch_size', 1)),
            gradient_clip=float(ft_config.get('gradient_clip', 1.0)),
            gradient_accumulation_steps=int(ft_config.get('gradient_accumulation_steps', 1)),
            patience=int(ft_config.get('patience', 5)),
            min_delta=float(ft_config.get('min_delta', 1e-4)),
            weight_decay=float(ft_config.get('weight_decay', 0.0)),
            dropout=float(ft_config.get('dropout', 0.0)),
            use_amp=bool(ft_config.get('use_amp', True)),
            val_batch_size=int(ft_config.get('val_batch_size', 1000)),
            eval_every_n_epochs=int(ft_config.get('eval_every_n_epochs', 1)),
            save_checkpoints=bool(ft_config.get('save_checkpoints', True)),
            checkpoint_dir=checkpoint_dir,
            device='cuda',
            random_state=int(self.config.get('experiment', {}).get('random_seed', 42)),
            max_data_size=int(ft_config.get('max_data_size', 150)),
        )

    def _create_finetuning_config_v2(self) -> FinetuningConfigV2:
        """Create FinetuningConfigV2 from experiment config."""
        ft_config = self.config.get('finetuning', {})

        return FinetuningConfigV2(
            learning_rate=float(ft_config.get('learning_rate', 1e-4)),
            weight_decay=float(ft_config.get('weight_decay', 0.0)),
            max_epochs=int(ft_config.get('max_epochs', 100)),
            patience=int(ft_config.get('patience', 16)),
            epoch_size=int(ft_config.get('epoch_size', 10)),
            seq_len_pred=int(ft_config.get('seq_len_pred', 1024)),
            max_context_size=int(ft_config['max_context_size']) if ft_config.get('max_context_size') is not None else None,
            batch_size=int(ft_config.get('batch_size', 1)),
            gradient_clip=float(ft_config.get('gradient_clip', 1.0)),
            use_amp=bool(ft_config.get('use_amp', True)),
            finetune_mode=str(ft_config.get('finetune_mode', 'full')),
            target_transform=ft_config.get('target_transform', None),
            checkpoint_path=ft_config.get('checkpoint_path', None),
            n_lr_warmup_epochs=int(ft_config.get('n_lr_warmup_epochs', 0)),
            softmax_temperature=float(ft_config.get('softmax_temperature', 0.9)),
            val_fraction=float(ft_config.get('val_fraction', 0.2)),
            eval_batch_size=int(ft_config.get('eval_batch_size', 4096)),
            device=str(ft_config.get('device', 'cuda')),
            random_state=int(self.config.get('experiment', {}).get('random_seed', 42)),
        )

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

        Override from base class to support fine-tuning with train/val split for TabPFN.

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

        if model_name == 'tabpfn':
            # For TabPFN fine-tuning, split train data into train+val
            val_fraction = self.config.get('finetuning', {}).get('val_fraction', 0.2)
            n_val = int(len(X_train) * val_fraction)
            n_train = len(X_train) - n_val

            # Use random split for train/val
            np.random.seed(self.config.get('experiment', {}).get('random_seed', 42))
            indices = np.random.permutation(len(X_train))
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            X_train_ft = X_train.iloc[train_idx]
            y_train_ft = y_train.iloc[train_idx]
            X_val_ft = X_train.iloc[val_idx]
            y_val_ft = y_train.iloc[val_idx]

            logger.info(f"  Fine-tuning with {len(X_train_ft)} train, {len(X_val_ft)} val samples")

            # Fit with validation data
            implementation = self.config.get('finetuning', {}).get('implementation', 'v1')
            if implementation == 'v2':
                # Pass continuous_cols from Phase2Preprocessor to v2 model
                continuous_cols = []
                if hasattr(self, 'phase2_preprocessor') and self.phase2_preprocessor is not None:
                    continuous_cols = self.phase2_preprocessor.continuous_cols
                model.fit(X_train_ft, y_train_ft, X_val_ft, y_val_ft,
                          continuous_cols=continuous_cols)
            else:
                model.fit(X_train_ft, y_train_ft, X_val_ft, y_val_ft)

            # Get training history
            history = model.get_training_history()
        else:
            # XGBoost: use all training data
            model.fit(X_train, y_train)
            history = None

        fit_time = time.time() - fit_start

        # Predict
        pred_start = time.time()
        if model_name == 'tabpfn':
            implementation = self.config.get('finetuning', {}).get('implementation', 'v1')
            if implementation == 'v2':
                # v2 uses training data stored during fit() as context by default
                logger.info(f"  Predicting with v2 model (using stored training data as context)")
                y_pred = model.predict(X_test)
            else:
                # v1: provide in-context examples from training data
                in_context_config = self.config.get('in_context', {})
                X_context, y_context = self._select_in_context_samples(
                    X_train, y_train, in_context_config
                )
                logger.info(f"  Providing {len(X_context)} in-context samples for TabPFN prediction")
                y_pred = model.predict(X_test, X_context=X_context, y_context=y_context)
        else:
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
            if hyperparams:
                if model_name == 'xgboost':
                    for key, value in hyperparams.items():
                        result[f'hyperparam_{key}'] = value
                    if hasattr(model, 'get_tune_time'):
                        result['tune_time'] = model.get_tune_time()
                elif model_name == 'tabpfn' and history:
                    # Add fine-tuning specific info
                    result['ft_best_epoch'] = history.best_epoch
                    result['ft_best_val_loss'] = history.best_val_loss
                    result['ft_n_epochs'] = len(history.train_losses)
                    for key, value in hyperparams.items():
                        result[f'hyperparam_{key}'] = value

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

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    def run_experiment(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Run the fine-tuning experiment.

        Automatically selects between new mode (test/train configs) and
        legacy mode (leave-one-out) based on config contents.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        if self.use_new_mode:
            return self.run_experiment_new_mode()
        else:
            return self.run_experiment_legacy()
