"""
Per-county scaling experiment: Train separate models per county with varying training set sizes.

This experiment trains individual models for each county using only that county's own
historical data, sweeping over training set sizes to build per-county learning curves.
It targets tiny (2-100 rows) and small (100-500 rows) counties from test_v1.

For each county x train_size x seed x model combination:
1. Sample train_size points from the county's training pool
2. Apply Phase 2 preprocessing (fit fresh each time)
3. Train and evaluate the model
4. Record metrics
"""

import pandas as pd
import numpy as np
import logging
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseExperimentRunner, ExperimentMetadata
from src.data import CleanedDataLoader
from src.data.split_strategies import (
    load_test_set_result,
    TestSetResult,
)
from src.data.preprocessing_utils import apply_phase2_preprocessing
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)

# Default exclude columns — same list used by get_train_test_data() in split_strategies.py
EXCLUDE_COLUMNS = [
    # ID/administrative columns
    "fips", "CLIP", "sale_date",
    # Additional administrative columns that may not have been dropped in Phase 1
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    # Baseline value - kept for evaluation but excluded from training features
    "CALCULATED_TOTAL_VALUE",
]


class PerCountyScalingExperiment(BaseExperimentRunner):
    """
    Experiment that trains separate models per county with varying training set sizes.

    Builds per-county learning curves by sweeping over training sizes and seeds.
    Targets tiny and small counties from test_v1.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.metadata = ExperimentMetadata(config)

        # Phase 2 preprocessing config
        self.phase2_config = config.get('preprocessing', {}).get('phase2_steps', {})

        # Data loader (no Phase 2 in loader — we apply it per combo)
        self.data_loader = CleanedDataLoader(
            cleaned_data_path=config['data']['cleaned_data_path'],
            target_column=config['data']['target_column'],
            phase2_config={}
        )

        # Override log_transformed based on metadata
        self.log_transformed = self.data_loader.is_target_log_transformed()
        logger.info(f"Target log-transformed: {self.log_transformed}")

        self.target_column = config['data']['target_column']

        # Experiment parameters
        self.target_buckets = config.get('target_buckets', ['tiny', 'small'])
        self.train_sizes = config.get('train_sizes', [5, 10, 20, 50, 100, 200])
        self.n_seeds = config.get('n_seeds', 3)
        self.base_seed = config.get('experiment', {}).get('random_seed', 42)

        # Test set directory
        self.test_set_dir = config['splits']['test_set_dir']

        # Checkpointing config
        self.checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_enabled = self.checkpoint_config.get('enabled', True)
        self.checkpoint_interval = self.checkpoint_config.get('interval', 50)
        self.checkpoint_resume = self.checkpoint_config.get('resume', True)

        # Single-county mode (for SLURM array jobs)
        self.single_county_fips = config.get('_single_county_fips', None)

    def create_model(self, model_name: str, train_size: int = None):
        """
        Create model with adaptive CV folds for small training sets.

        Overrides base class to reduce XGBoost Optuna CV folds when
        training set is very small (< 30 samples).
        """
        if model_name == 'xgboost' and train_size is not None:
            xgboost_config = self.config.get('xgboost', {})
            cv_folds = xgboost_config.get('optuna_cv_folds', 3)

            # Reduce CV folds for very small training sets
            if train_size < 30:
                cv_folds = min(cv_folds, 2)
                logger.debug(f"Reduced CV folds to {cv_folds} for train_size={train_size}")

            from src.models import XGBoostModel
            model = XGBoostModel(
                n_trials=xgboost_config.get('optuna_trials', 50),
                cv_folds=cv_folds,
                use_gpu=xgboost_config.get('use_gpu', True),
                random_state=self.config.get('experiment', {}).get('random_seed', 42)
            )
            return model
        else:
            return super().create_model(model_name)

    def _load_data_and_group_by_county(self) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        Load test set, load data by indices, remap, and group by county.

        Returns:
            Tuple of (df, county_data) where county_data maps fips -> {
                'train_pool_indices': array of remapped indices,
                'test_indices': array of remapped indices,
                'size_bucket': str,
                'county_info': dict,
            }
        """
        logger.info("Loading pre-generated test set...")
        test_result = load_test_set_result(self.test_set_dir)

        # Load only test + train_pool indices (same pattern as cross_county.py)
        logger.info("Loading data for pre-generated splits (memory-efficient)...")
        all_indices = np.concatenate([
            test_result.test_indices,
            test_result.train_pool_indices,
        ])
        unique_indices = np.unique(all_indices)
        df = self.data_loader.load_data_by_indices(unique_indices)
        logger.info(f"  Loaded {len(df):,} rows")

        # Create index mapping: old_index -> new_index in the subset DataFrame
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

        # Remap indices
        remapped_test = np.array([index_map[idx] for idx in test_result.test_indices])
        remapped_train_pool = np.array([index_map[idx] for idx in test_result.train_pool_indices])

        # Reset DataFrame index
        df = df.reset_index(drop=True)

        # Group by county
        fips_values = df['fips'].values

        # Build per-county index sets
        county_data = {}
        for fips in test_result.test_counties:
            county_info = test_result.county_info.get(fips, {})
            bucket = county_info.get('size_bucket', 'unknown')

            # Filter to target buckets
            if bucket not in self.target_buckets:
                continue

            # Get this county's train pool and test indices
            train_mask = fips_values[remapped_train_pool] == fips
            test_mask = fips_values[remapped_test] == fips

            county_train_pool = remapped_train_pool[train_mask]
            county_test = remapped_test[test_mask]

            if len(county_train_pool) == 0 or len(county_test) == 0:
                logger.warning(f"Skipping county {fips}: train_pool={len(county_train_pool)}, test={len(county_test)}")
                continue

            county_data[fips] = {
                'train_pool_indices': county_train_pool,
                'test_indices': county_test,
                'size_bucket': bucket,
                'county_info': county_info,
            }

        logger.info(f"Target counties: {len(county_data)} ({', '.join(self.target_buckets)})")
        for bucket in self.target_buckets:
            count = sum(1 for c in county_data.values() if c['size_bucket'] == bucket)
            logger.info(f"  {bucket}: {count} counties")

        return df, county_data

    def _get_features_and_target(
        self,
        df: pd.DataFrame,
        indices: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X and y for given indices, dropping exclude columns."""
        subset = df.iloc[indices]
        y = subset[self.target_column]
        X = subset.drop(
            columns=[self.target_column] + [c for c in EXCLUDE_COLUMNS if c in subset.columns]
        )
        return X, y

    def _load_checkpoint(self, results_dir: Path) -> Tuple[List[Dict], set]:
        """Load checkpoint if it exists and resume is enabled."""
        results = []
        completed_keys = set()

        if not self.checkpoint_resume or not self.checkpoint_enabled:
            return results, completed_keys

        checkpoint_csv = results_dir / 'results_checkpoint.csv'
        checkpoint_keys = results_dir / 'completed_keys.pkl'

        if checkpoint_csv.exists() and checkpoint_keys.exists():
            logger.info("Resuming from checkpoint...")
            results_df = pd.read_csv(checkpoint_csv)
            results = results_df.to_dict('records')

            with open(checkpoint_keys, 'rb') as f:
                completed_keys = pickle.load(f)

            logger.info(f"  Loaded {len(results)} results, {len(completed_keys)} completed keys")

        return results, completed_keys

    def _save_checkpoint(self, results: List[Dict], completed_keys: set, results_dir: Path):
        """Save checkpoint to disk."""
        if not self.checkpoint_enabled or len(results) == 0:
            return

        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        pd.DataFrame(results).to_csv(results_dir / 'results_checkpoint.csv', index=False)

        # Save completed keys
        with open(results_dir / 'completed_keys.pkl', 'wb') as f:
            pickle.dump(completed_keys, f)

        logger.debug(f"Checkpoint saved: {len(results)} results")

    def run_experiment(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """Run the per-county scaling experiment."""
        logger.info("=" * 80)
        logger.info(f"PER-COUNTY SCALING EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("=" * 80)

        # Load data and group by county
        df, county_data = self._load_data_and_group_by_county()

        # Filter to single county if in SLURM array mode
        if self.single_county_fips is not None:
            fips = self.single_county_fips
            if fips not in county_data:
                logger.error(f"County {fips} not found in target counties. Available: {sorted(county_data.keys())}")
                return pd.DataFrame(), None, None
            county_data = {fips: county_data[fips]}
            logger.info(f"Single-county mode: processing only FIPS {fips}")

        # Setup output
        results_dir = Path(self.config['output']['results_dir'])
        if self.single_county_fips is not None:
            results_dir = results_dir / f"county_{self.single_county_fips}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        all_results, completed_keys = self._load_checkpoint(results_dir)

        # Get enabled models
        enabled_models = self.get_enabled_models()

        # Build combo list
        combos = []
        for fips, cdata in sorted(county_data.items()):
            pool_size = len(cdata['train_pool_indices'])
            for train_size in self.train_sizes:
                # Skip if requested size exceeds pool
                if train_size > pool_size:
                    continue
                for seed_idx in range(self.n_seeds):
                    seed = self.base_seed + seed_idx
                    for model_name in enabled_models:
                        key = (fips, train_size, seed, model_name)
                        if key not in completed_keys:
                            combos.append((fips, train_size, seed, model_name))

        total_combos = len(combos) + len(completed_keys)
        logger.info(f"Total combos: {total_combos} ({len(completed_keys)} already done, {len(combos)} remaining)")
        logger.info(f"Models: {enabled_models}")
        logger.info(f"Train sizes: {self.train_sizes}")
        logger.info(f"Seeds: {self.n_seeds}")
        logger.info("=" * 80)

        start_time = time.time()
        combos_since_checkpoint = 0

        for combo_idx, (fips, train_size, seed, model_name) in enumerate(combos):
            cdata = county_data[fips]
            key = (fips, train_size, seed, model_name)

            logger.info(
                f"[{combo_idx + 1}/{len(combos)}] "
                f"FIPS={fips} size={train_size} seed={seed} model={model_name}"
            )

            try:
                # Sample training indices
                rng = np.random.RandomState(seed)
                train_pool = cdata['train_pool_indices']
                sampled_train_idx = rng.choice(train_pool, size=train_size, replace=False)

                # Extract features and targets
                X_train, y_train = self._get_features_and_target(df, sampled_train_idx)
                X_test, y_test = self._get_features_and_target(df, cdata['test_indices'])

                # Apply Phase 2 preprocessing (fit fresh each time)
                if self.phase2_config:
                    X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        config=self.phase2_config
                    )

                # Train and predict — use adaptive create_model for XGB
                model = self.create_model(model_name, train_size=train_size)

                fit_start = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - fit_start

                pred_start = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - pred_start

                metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)

                model.cleanup()

                result = {
                    'fips': fips,
                    'size_bucket': cdata['size_bucket'],
                    'county_train_pool_size': len(train_pool),
                    'county_test_size': len(cdata['test_indices']),
                    'requested_train_size': train_size,
                    'actual_train_size': len(X_train),
                    'seed': seed,
                    'model': model_name,
                    'n_features': X_train.shape[1],
                    'fit_time': fit_time,
                    'pred_time': pred_time,
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'mse': metrics['mse'],
                    'status': 'success',
                }

                logger.info(
                    f"  R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} "
                    f"fit={fit_time:.1f}s pred={pred_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"  Failed: {e}", exc_info=True)
                result = {
                    'fips': fips,
                    'size_bucket': cdata['size_bucket'],
                    'county_train_pool_size': len(cdata['train_pool_indices']),
                    'county_test_size': len(cdata['test_indices']),
                    'requested_train_size': train_size,
                    'actual_train_size': 0,
                    'seed': seed,
                    'model': model_name,
                    'n_features': 0,
                    'fit_time': 0,
                    'pred_time': 0,
                    'r2': np.nan,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mape': np.nan,
                    'mse': np.nan,
                    'status': f'failed: {str(e)}',
                }

            result = self.metadata.add_to_result(result)
            all_results.append(result)
            completed_keys.add(key)
            combos_since_checkpoint += 1

            # Periodic checkpoint
            if combos_since_checkpoint >= self.checkpoint_interval:
                self._save_checkpoint(all_results, completed_keys, results_dir)
                combos_since_checkpoint = 0

        # Final save
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()

        if not results_df.empty:
            self.save_results(results_df, results_dir, 'results.csv')

        # Also save final checkpoint
        self._save_checkpoint(all_results, completed_keys, results_dir)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Total results: {len(all_results)}")
        n_success = sum(1 for r in all_results if r.get('status') == 'success')
        logger.info(f"Successful: {n_success}")
        logger.info(f"Failed: {len(all_results) - n_success}")

        return results_df, None, None
