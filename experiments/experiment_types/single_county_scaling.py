"""
Single-county data scaling experiment: Learning curves for one county.

For a given county (e.g., Cook County, FIPS 17031):
1. Load all county data from the preprocessed parquet
2. Split temporally: oldest 80% = train pool, most recent 20% = test
3. For each (train_size, seed) combination:
   a. Sample train_size rows from the train pool
   b. Apply Phase 2 preprocessing (fit on sample, apply to both)
   c. Train all enabled models
   d. Evaluate on the fixed test set
   e. Record metrics

This produces learning curves showing how model performance scales with
training data size, which is the core data scarcity comparison.
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseExperimentRunner, ExperimentMetadata
from src.data import CleanedDataLoader
from src.data.preprocessing_utils import apply_phase2_preprocessing
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)

# Columns to exclude from features (IDs, leakage, etc.)
EXCLUDE_COLUMNS = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]


class SingleCountyScalingExperiment(BaseExperimentRunner):
    """
    Experiment that trains models on varying amounts of single-county data
    to build learning curves.
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
        logger.info(f"Target log-transformed: {self.log_transformed}")

        self.target_column = config['data']['target_column']

        # County config
        scaling_config = config.get('single_county_scaling', {})
        self.county_fips = scaling_config['county_fips']
        self.train_sizes = scaling_config['train_sizes']
        self.seeds = scaling_config['seeds']
        self.test_fraction = scaling_config.get('test_fraction', 0.2)
        self.temporal_split = scaling_config.get('temporal_split', True)
        self.temporal_column = scaling_config.get('temporal_column', 'sale_day')

        # Ratio filter
        ratio_filter_config = config.get('ratio_filter', {})
        self.ratio_filter_enabled = ratio_filter_config.get('enabled', False)
        self.ratio_filter_drop_bottom_percentile = ratio_filter_config.get('drop_bottom_percentile', 0)
        self.ratio_filter_drop_top_percentile = ratio_filter_config.get('drop_top_percentile', 0)
        self.ratio_filter_by_sale_year = ratio_filter_config.get('by_sale_year', True)

        # Intermediate saving
        self.save_interval = scaling_config.get('save_interval', 20)

        # Output
        self.output_dir = Path(config['output']['results_dir'])

    def _load_county_data(self) -> pd.DataFrame:
        """Load all rows for the target county from the preprocessed parquet."""
        logger.info(f"Loading data for county FIPS {self.county_fips}")

        # Load full dataset and filter to county
        df = self.data_loader.load_data()
        county_df = df[df['fips'] == self.county_fips].copy()

        if len(county_df) == 0:
            raise ValueError(f"County FIPS {self.county_fips} not found in preprocessed data")

        logger.info(f"  County {self.county_fips}: {len(county_df):,} rows")

        # Clear the full dataset from cache to free memory
        self.data_loader.clear_cache()

        return county_df

    def _apply_ratio_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply assessed-to-sale ratio filter if enabled."""
        if not self.ratio_filter_enabled:
            return df

        before = len(df)

        if 'MARKET_TOTAL_VALUE' not in df.columns or self.target_column not in df.columns:
            logger.warning("Ratio filter: required columns not found, skipping")
            return df

        # Compute ratio (target is log-transformed, so exp first)
        if self.log_transformed:
            ratio = df['MARKET_TOTAL_VALUE'] / np.exp(df[self.target_column])
        else:
            ratio = df['MARKET_TOTAL_VALUE'] / df[self.target_column]

        if self.ratio_filter_by_sale_year and 'sale_year' in df.columns:
            # Percentile rank within each year
            pct_rank = df.groupby('sale_year').apply(
                lambda g: ratio.loc[g.index].rank(pct=True) * 100
            )
            if hasattr(pct_rank, 'droplevel'):
                pct_rank = pct_rank.droplevel(0).sort_index()
        else:
            pct_rank = ratio.rank(pct=True) * 100

        mask = pd.Series(True, index=df.index)
        if self.ratio_filter_drop_bottom_percentile > 0:
            mask &= pct_rank >= self.ratio_filter_drop_bottom_percentile
        if self.ratio_filter_drop_top_percentile > 0:
            mask &= pct_rank <= (100 - self.ratio_filter_drop_top_percentile)

        df = df[mask].reset_index(drop=True)
        logger.info(f"  Ratio filter: {before:,} -> {len(df):,} rows")

        return df

    def _split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split county data into train pool and test set."""
        if self.temporal_split:
            if self.temporal_column not in df.columns:
                raise ValueError(
                    f"Temporal column '{self.temporal_column}' not found. "
                    f"Available columns: {sorted(df.columns.tolist())}"
                )
            # Sort by time, take most recent fraction as test
            df = df.sort_values(self.temporal_column).reset_index(drop=True)
            split_idx = int(len(df) * (1 - self.test_fraction))
            train_pool = df.iloc[:split_idx].copy()
            test_set = df.iloc[split_idx:].copy()
            logger.info(
                f"  Temporal split: train_pool={len(train_pool):,}, "
                f"test_set={len(test_set):,}"
            )
        else:
            # Random split with fixed seed
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            split_idx = int(len(df) * (1 - self.test_fraction))
            train_pool = df.iloc[indices[:split_idx]].copy()
            test_set = df.iloc[indices[split_idx:]].copy()
            logger.info(
                f"  Random split: train_pool={len(train_pool):,}, "
                f"test_set={len(test_set):,}"
            )

        return train_pool, test_set

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target, dropping excluded columns."""
        exclude = set(EXCLUDE_COLUMNS)
        feature_cols = [c for c in df.columns if c not in exclude and c != self.target_column]
        X = df[feature_cols].copy()
        y = df[self.target_column].copy()
        return X, y

    def run_experiment(self):
        """
        Run the single-county data scaling experiment.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        # Load and prepare county data
        county_df = self._load_county_data()
        county_df = self._apply_ratio_filter(county_df)
        train_pool_df, test_df = self._split_train_test(county_df)

        # Extract features/target for test set (fixed across all trials)
        X_test_raw, y_test_raw = self._prepare_features(test_df)

        # Get train pool features
        X_train_pool_raw, y_train_pool_raw = self._prepare_features(train_pool_df)

        # Get continuous columns for Phase 2
        continuous_cols = self.data_loader.get_continuous_cols()
        continuous_cols = [c for c in continuous_cols if c in X_train_pool_raw.columns]

        # Get enabled models
        enabled_models = self.get_enabled_models()
        logger.info(f"Enabled models: {enabled_models}")

        # Filter train_sizes to feasible values
        max_train_size = len(X_train_pool_raw)
        feasible_train_sizes = [s for s in self.train_sizes if s <= max_train_size]
        if len(feasible_train_sizes) < len(self.train_sizes):
            skipped = [s for s in self.train_sizes if s > max_train_size]
            logger.warning(
                f"Train pool has {max_train_size} rows. "
                f"Skipping train_sizes > {max_train_size}: {skipped}"
            )

        total_trials = len(feasible_train_sizes) * len(self.seeds) * len(enabled_models)
        logger.info(
            f"Running {total_trials} trials: "
            f"{len(feasible_train_sizes)} sizes x {len(self.seeds)} seeds x "
            f"{len(enabled_models)} models"
        )

        # Run experiment loop
        results = []
        trial_count = 0
        experiment_start = time.time()

        for seed in self.seeds:
            for train_size in feasible_train_sizes:
                # Sample from train pool
                rng = np.random.RandomState(seed)
                sample_indices = rng.choice(
                    len(X_train_pool_raw), size=train_size, replace=False
                )
                X_train_sample = X_train_pool_raw.iloc[sample_indices].reset_index(drop=True)
                y_train_sample = y_train_pool_raw.iloc[sample_indices].reset_index(drop=True)

                # Apply Phase 2 preprocessing (fit on this sample)
                if self.phase2_config:
                    X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                        X_train_sample, y_train_sample,
                        X_test_raw.copy(), y_test_raw.copy(),
                        config=self.phase2_config,
                        continuous_cols=continuous_cols,
                    )
                else:
                    X_train = X_train_sample
                    y_train = y_train_sample
                    X_test = X_test_raw.copy()
                    y_test = y_test_raw.copy()

                for model_name in enabled_models:
                    trial_count += 1

                    try:
                        result, _, _ = self.train_and_predict(
                            model_name=model_name,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                        )

                        # Add experiment metadata
                        result['seed'] = seed
                        result['train_size'] = train_size
                        result['county_fips'] = self.county_fips
                        result['n_test'] = len(X_test)
                        result['status'] = 'success'

                        results.append(result)

                        logger.info(
                            f"  [{trial_count}/{total_trials}] "
                            f"seed={seed}, train_size={train_size}, "
                            f"model={model_name}: "
                            f"R2={result['r2']:.4f}, MAE={result['mae']:.2f}"
                        )

                    except Exception as e:
                        logger.error(
                            f"  [{trial_count}/{total_trials}] "
                            f"FAILED seed={seed}, train_size={train_size}, "
                            f"model={model_name}: {e}"
                        )
                        results.append({
                            'model': model_name,
                            'seed': seed,
                            'train_size': train_size,
                            'county_fips': self.county_fips,
                            'n_test': len(X_test),
                            'status': f'error: {e}',
                        })

                # Intermediate save
                if trial_count % (self.save_interval * len(enabled_models)) == 0:
                    self._save_intermediate(results)

        # Final save
        total_time = time.time() - experiment_start
        logger.info(f"Experiment complete: {trial_count} trials in {total_time:.1f}s")

        results_df = pd.DataFrame(results)
        self.save_results(results_df, self.output_dir, 'results.csv')

        n_success = len([r for r in results if r.get('status') == 'success'])
        n_failed = len(results) - n_success
        logger.info(f"  Successes: {n_success}, Failures: {n_failed}")

        return results_df, None, None

    def _save_intermediate(self, results: List[Dict]):
        """Save intermediate results for monitoring/recovery."""
        if not results:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        intermediate_file = self.output_dir / 'results_intermediate.csv'
        df.to_csv(intermediate_file, index=False)
        logger.info(f"  Intermediate results saved ({len(results)} rows)")
