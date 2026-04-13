"""
Global finetuning experiment: Finetune TabPFN once on a large pooled dataset.

Two variants:
  (a) External: Finetune on non-test_v4 counties (no test data leakage)
  (b) Internal: Finetune on test_v4 train pool indices (county historical data)

The resulting checkpoint is saved to disk and can be loaded by the
geo_pooling experiment as a drop-in replacement for zero-shot TabPFN.

Usage:
  python experiments/run_experiment.py --config experiments/configs/global_finetuning/...yaml
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseExperimentRunner, ExperimentMetadata
from src.data import CleanedDataLoader
from src.data.split_strategies import load_test_set_result, TestSetResult
from src.data.preprocessing_utils import Phase2Preprocessor
from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel, FinetuningConfigV2

logger = logging.getLogger(__name__)

# Default exclude columns (same list as geo_pooling.py)
EXCLUDE_COLUMNS = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]


class GlobalFinetuningExperiment(BaseExperimentRunner):
    """
    Finetune TabPFN on a large pooled dataset and save the checkpoint.

    Supports two variants:
    - "external": train on non-test_v4 county data (no test leakage)
    - "internal": train on test_v4 train pool data (historical county data)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.metadata = ExperimentMetadata(config)

        # Phase 2 preprocessing config
        self.phase2_config = config.get('preprocessing', {}).get('phase2_steps', {})

        # Data loader
        self.data_loader = CleanedDataLoader(
            cleaned_data_path=config['data']['cleaned_data_path'],
            target_column=config['data']['target_column'],
            phase2_config={}
        )

        self.log_transformed = self.data_loader.is_target_log_transformed()
        self.target_column = config['data']['target_column']

        # Global finetuning config
        gft_config = config.get('global_finetuning', {})
        self.variant = gft_config.get('variant', 'external')  # "external" or "internal"
        self.n_samples = gft_config.get('n_samples', 15000)
        self.sampling_strategy = gft_config.get('sampling_strategy', 'uniform')

        # Test set directory (needed to identify test_v4 counties)
        self.test_set_dir = config['splits']['test_set_dir']

        # Ratio filter
        ratio_filter_config = config.get('ratio_filter', {})
        self.ratio_filter_enabled = ratio_filter_config.get('enabled', False)
        self.ratio_filter_drop_bottom_percentile = ratio_filter_config.get('drop_bottom_percentile', 0)
        self.ratio_filter_drop_top_percentile = ratio_filter_config.get('drop_top_percentile', 0)
        self.ratio_filter_by_sale_year = ratio_filter_config.get('by_sale_year', True)

        # Finetuning config
        self.finetuning_config = config.get('finetuning', {})

        # Output
        self.checkpoint_dir = config['output'].get('checkpoint_dir',
                                                    config['output']['results_dir'])

    def _load_training_data(self, test_result: TestSetResult) -> pd.DataFrame:
        """Load training data based on the variant."""
        import pyarrow.parquet as pq

        if self.variant == 'external':
            return self._load_external_variant(test_result)
        elif self.variant == 'internal':
            return self._load_internal_variant(test_result)
        else:
            raise ValueError(f"Unknown variant: {self.variant}. Use 'external' or 'internal'.")

    def _load_external_variant(self, test_result: TestSetResult) -> pd.DataFrame:
        """Load data from non-test_v4 counties."""
        logger.info("Loading training data: EXTERNAL variant (non-test_v4 counties)")

        # Step 1: Read fips column to find non-test_v4 rows
        all_fips = self.data_loader.load_fips_column()
        test_v4_fips = set(test_result.test_counties)

        # Build mask of non-test_v4 rows
        external_mask = np.array([int(f) not in test_v4_fips for f in all_fips])
        external_indices = np.where(external_mask)[0]
        logger.info(f"  Non-test_v4 rows: {len(external_indices):,} out of {len(all_fips):,}")

        # Step 2: Sample indices
        rng = np.random.RandomState(
            self.config.get('experiment', {}).get('random_seed', 42)
        )

        if self.sampling_strategy == 'uniform':
            n_take = min(self.n_samples, len(external_indices))
            sampled_indices = rng.choice(external_indices, size=n_take, replace=False)
        elif self.sampling_strategy == 'stratified_by_county':
            # Sample roughly equal numbers from each county
            external_fips_vals = all_fips[external_indices]
            unique_ext_fips = np.unique(external_fips_vals)
            per_county = max(1, self.n_samples // len(unique_ext_fips))
            sampled = []
            for fips_val in unique_ext_fips:
                county_rows = external_indices[external_fips_vals == fips_val]
                n_take = min(per_county, len(county_rows))
                sampled.extend(rng.choice(county_rows, size=n_take, replace=False))
                if len(sampled) >= self.n_samples:
                    break
            sampled_indices = np.array(sampled[:self.n_samples])
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

        sampled_indices = np.sort(sampled_indices)
        logger.info(f"  Sampled {len(sampled_indices):,} rows ({self.sampling_strategy})")

        # Step 3: Load the sampled data
        df = self.data_loader.load_data_by_indices(sampled_indices)
        return df.reset_index(drop=True)

    def _load_internal_variant(self, test_result: TestSetResult) -> pd.DataFrame:
        """Load data from test_v4 train pool indices."""
        logger.info("Loading training data: INTERNAL variant (test_v4 train pool)")

        train_pool_indices = test_result.train_pool_indices
        logger.info(f"  test_v4 train pool: {len(train_pool_indices):,} rows")

        # Sample if needed
        rng = np.random.RandomState(
            self.config.get('experiment', {}).get('random_seed', 42)
        )
        if len(train_pool_indices) > self.n_samples:
            sampled_indices = rng.choice(
                train_pool_indices, size=self.n_samples, replace=False
            )
        else:
            sampled_indices = train_pool_indices

        sampled_indices = np.sort(sampled_indices)
        logger.info(f"  Sampled {len(sampled_indices):,} rows")

        df = self.data_loader.load_data_by_indices(sampled_indices)
        return df.reset_index(drop=True)

    def _apply_ratio_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ratio filter to training data."""
        if not self.ratio_filter_enabled:
            return df

        if 'MARKET_TOTAL_VALUE' not in df.columns:
            logger.warning("ratio_filter enabled but MARKET_TOTAL_VALUE not found, skipping")
            return df

        ratio = df['MARKET_TOTAL_VALUE'] / np.exp(df[self.target_column])

        if self.ratio_filter_by_sale_year and 'sale_year' in df.columns:
            def _pct_rank(g):
                return g.rank(pct=True) * 100
            ratio_pct = ratio.groupby(df['sale_year']).transform(_pct_rank)
        else:
            ratio_pct = ratio.rank(pct=True) * 100

        keep = np.ones(len(df), dtype=bool)
        if self.ratio_filter_drop_bottom_percentile > 0:
            keep &= ratio_pct.values >= self.ratio_filter_drop_bottom_percentile
        if self.ratio_filter_drop_top_percentile > 0:
            keep &= ratio_pct.values <= (100 - self.ratio_filter_drop_top_percentile)

        n_before = len(df)
        df = df[keep].reset_index(drop=True)
        logger.info(
            f"  Ratio filter: {n_before:,} -> {len(df):,} rows "
            f"(dropped {n_before - len(df):,})"
        )
        return df

    def _get_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X and y, dropping exclude columns."""
        y = df[self.target_column]
        X = df.drop(
            columns=[self.target_column] + [c for c in EXCLUDE_COLUMNS if c in df.columns]
        )
        return X, y

    def run_experiment(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """Run the global finetuning experiment."""
        logger.info("=" * 80)
        logger.info(f"GLOBAL FINETUNING EXPERIMENT: {self.config['experiment']['name']}")
        logger.info(f"  Variant: {self.variant}")
        logger.info(f"  Target samples: {self.n_samples}")
        logger.info(f"  Sampling strategy: {self.sampling_strategy}")
        logger.info("=" * 80)

        start_time = time.time()

        # Load test set info (to know which counties are test_v4)
        logger.info("Loading test set info...")
        test_result = load_test_set_result(self.test_set_dir)

        # Load training data
        df = self._load_training_data(test_result)

        # Apply ratio filter
        df = self._apply_ratio_filter(df)

        # Extract county IDs before dropping FIPS from features
        county_ids = df['fips'].copy() if 'fips' in df.columns else None

        # Extract features and target
        X, y = self._get_features_and_target(df)

        # Apply Phase 2 preprocessing.
        # For per_county training mode: skip StandardScaler — normalization is applied
        # per-county inside _fit_per_county() to match the per-county scaling done at
        # inference time in geo_pooling. Winsorization and imputation are still applied
        # globally here so the tensors are clean before the training loop.
        if self.phase2_config:
            logger.info("Applying Phase 2 preprocessing...")
            phase2_config_for_fit = dict(self.phase2_config)
            if self.finetuning_config.get('training_mode', 'global') == 'per_county':
                phase2_config_for_fit['normalize_continuous'] = False
                logger.info("  (Skipping StandardScaler: per_county mode normalizes per step)")
            phase2_preprocessor = Phase2Preprocessor(phase2_config_for_fit)
            phase2_preprocessor.fit(X, y)
            X = phase2_preprocessor.transform(X)
            y = phase2_preprocessor.transform_target(y)

        logger.info(f"Final training data: {X.shape[0]:,} samples, {X.shape[1]} features")

        # Build finetuning config
        ft = self.finetuning_config
        seed = self.config.get('experiment', {}).get('random_seed', 42)
        finetune_cfg = FinetuningConfigV2(
            learning_rate=float(ft.get('learning_rate', 1e-4)),
            weight_decay=float(ft.get('weight_decay', 0.0)),
            max_epochs=int(ft.get('max_epochs', 100)),
            patience=int(ft.get('patience', 16)),
            epoch_size=int(ft.get('epoch_size', 10)),
            seq_len_pred=int(ft.get('seq_len_pred', 1024)),
            max_context_size=int(ft['max_context_size']) if ft.get('max_context_size') is not None else None,
            min_context_size=int(ft.get('min_context_size', 5)),
            batch_size=int(ft.get('batch_size', 1)),
            gradient_clip=float(ft.get('gradient_clip', 1.0)),
            use_amp=bool(ft.get('use_amp', True)),
            finetune_mode=str(ft.get('finetune_mode', 'full')),
            lora_rank=int(ft.get('lora_rank', 0)),
            lora_alpha=float(ft.get('lora_alpha', 16.0)),
            target_transform=ft.get('target_transform', None),
            checkpoint_path=ft.get('checkpoint_path', None),
            n_lr_warmup_epochs=int(ft.get('n_lr_warmup_epochs', 0)),
            softmax_temperature=float(ft.get('softmax_temperature', 0.9)),
            val_fraction=float(ft.get('val_fraction', 0.2)),
            eval_batch_size=int(ft.get('eval_batch_size', 4096)),
            device=str(ft.get('device', 'cuda')),
            random_state=seed,
            training_mode=str(ft.get('training_mode', 'global')),
            min_county_size=int(ft.get('min_county_size', 5)),
            context_fraction_range=tuple(ft.get('context_fraction_range', [0.3, 0.7])),
            spike_diagnostics=bool(ft.get('spike_diagnostics', False)),
            spike_threshold=float(ft.get('spike_threshold', 100.0)),
        )

        # Train
        logger.info("Starting finetuning...")
        model = DirectFineTunedTabPFNModel(finetune_cfg)
        model.fit(X, y, county_ids=county_ids)

        # Save checkpoint
        logger.info(f"Saving checkpoint to {self.checkpoint_dir}")
        model.save_to_disk(self.checkpoint_dir)

        # Build results
        total_time = time.time() - start_time
        history = model.get_training_history()

        result = {
            'experiment_name': self.config['experiment']['name'],
            'variant': self.variant,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'best_epoch': history.best_epoch,
            'best_val_loss': history.best_val_loss,
            'n_epochs_trained': len(history.train_losses),
            'total_time_seconds': total_time,
            'checkpoint_dir': str(self.checkpoint_dir),
            'status': 'success',
        }
        result = self.metadata.add_to_result(result)

        # Save results
        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame([result])
        results_df.to_csv(results_dir / 'results.csv', index=False)

        # Save training history separately for easy plotting
        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history.to_dict(), f, indent=2)

        # Cleanup
        model.cleanup()

        logger.info("=" * 80)
        logger.info("GLOBAL FINETUNING COMPLETE")
        logger.info(f"  Best epoch: {history.best_epoch}")
        logger.info(f"  Total time: {total_time / 60:.2f} minutes")
        logger.info(f"  Checkpoint: {self.checkpoint_dir}")
        logger.info("=" * 80)

        return results_df, None, None
