"""
Geographic pooling experiment: Per-county models trained on local + neighbor data.

For each test county:
1. Take the county's own historical data
2. Find geographically nearest neighbor counties
3. Sample data from neighbors up to a configurable budget
4. Train model(s) on this county-specific pooled dataset
5. Evaluate on the county's test set

Supports checkpointing and SLURM array jobs (--county_index).
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
from src.data.geo_utils import load_centroids, get_neighbors
from src.evaluation import compute_metrics
from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel, FinetuningConfigV2

logger = logging.getLogger(__name__)

# Default exclude columns — same list used by get_train_test_data() in split_strategies.py
EXCLUDE_COLUMNS = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]


class GeoPoolingExperiment(BaseExperimentRunner):
    """
    Experiment that trains per-county models using geographically pooled data.
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

        # Test set directory
        self.test_set_dir = config['splits']['test_set_dir']

        # Bucket filtering
        self.target_buckets = config.get('filter_buckets', ['tiny', 'small', 'medium'])

        # Geo pooling parameters
        geo_config = config.get('geo_pooling', {})
        self.centroids_csv = geo_config.get('centroids_csv', 'data/us_county_latlng.csv')
        self.max_k_neighbors = geo_config.get('max_k_neighbors', None)
        self.max_distance_miles = geo_config.get('max_distance_miles', None)
        # Budget: ratio of neighbor samples to own samples, OR a fixed count
        self.neighbor_budget_ratio = geo_config.get('neighbor_budget_ratio', None)
        self.neighbor_budget_fixed = geo_config.get('neighbor_budget_fixed', None)
        self.max_samples_per_neighbor = geo_config.get('max_samples_per_neighbor', None)
        self.max_total_training_size = geo_config.get('max_total_training_size', None)
        self.restrict_neighbors_to_same_size_bucket = geo_config.get('restrict_neighbors_to_same_size_bucket', False)
        self.extend_neighbor_pool_beyond_testv4 = geo_config.get('extend_neighbor_pool_beyond_testv4', False)
        self.diversify_neighbors = geo_config.get('diversify_neighbors', False)
        self.diversify_n = geo_config.get('diversify_n', 3)

        if self.neighbor_budget_ratio is None and self.neighbor_budget_fixed is None:
            raise ValueError("geo_pooling config must specify neighbor_budget_ratio or neighbor_budget_fixed")

        # Checkpointing
        checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', True)
        self.checkpoint_interval = checkpoint_config.get('interval', 10)
        self.checkpoint_resume = checkpoint_config.get('resume', False)

        # Ratio filter at load time (applied before index mapping)
        # Ratio = MARKET_TOTAL_VALUE / exp(SALE_AMOUNT) (target is log-transformed)
        ratio_filter_config = config.get('ratio_filter', {})
        self.ratio_filter_enabled = ratio_filter_config.get('enabled', False)
        self.ratio_filter_drop_bottom_percentile = ratio_filter_config.get('drop_bottom_percentile', 0)
        self.ratio_filter_drop_top_percentile = ratio_filter_config.get('drop_top_percentile', 0)
        self.ratio_filter_by_sale_year = ratio_filter_config.get('by_sale_year', True)

        # Finetuning config (for tabpfn_finetuned model)
        self.finetuning_config = config.get('finetuning', {})
        self.min_finetune_size = self.finetuning_config.get('min_train_size', 20)

        # SLURM array job support: process a subset of counties
        self.county_index = config.get('_county_index', None)
        self.county_chunk_size = config.get('_county_chunk_size', None)
        self.n_chunks = config.get('_n_chunks', None)

    def create_model(self, model_name: str, train_size: int = None):
        """Create model with adaptive CV folds for small training sets."""
        if model_name == 'tabpfn_finetuned':
            ft = self.finetuning_config
            seed = self.config.get('experiment', {}).get('random_seed', 42)
            finetune_cfg = FinetuningConfigV2(
                learning_rate=float(ft.get('learning_rate', 1e-4)),
                weight_decay=float(ft.get('weight_decay', 0.0)),
                max_epochs=int(ft.get('max_epochs', 50)),
                patience=int(ft.get('patience', 8)),
                epoch_size=int(ft.get('epoch_size', 10)),
                seq_len_pred=int(ft.get('seq_len_pred', 128)),
                max_context_size=int(ft['max_context_size']) if ft.get('max_context_size') is not None else None,
                batch_size=int(ft.get('batch_size', 1)),
                gradient_clip=float(ft.get('gradient_clip', 1.0)),
                use_amp=bool(ft.get('use_amp', True)),
                finetune_mode=str(ft.get('finetune_mode', 'full')),
                target_transform=ft.get('target_transform', None),
                checkpoint_path=ft.get('checkpoint_path', None),
                n_lr_warmup_epochs=int(ft.get('n_lr_warmup_epochs', 0)),
                softmax_temperature=float(ft.get('softmax_temperature', 0.9)),
                val_fraction=float(ft.get('val_fraction', 0.2)),
                eval_batch_size=int(ft.get('eval_batch_size', 4096)),
                device=str(ft.get('device', 'cuda')),
                random_state=seed,
            )
            return DirectFineTunedTabPFNModel(finetune_cfg)

        if model_name == 'tabpfn_global_finetuned':
            # Load a pre-finetuned TabPFN model from a saved checkpoint
            checkpoint_dir = self._get_global_finetuned_checkpoint_dir()
            model = DirectFineTunedTabPFNModel.load_from_disk(checkpoint_dir)
            return model

        if model_name == 'xgboost' and train_size is not None:
            xgboost_config = self.config.get('xgboost', {})
            cv_folds = xgboost_config.get('optuna_cv_folds', 3)

            # Cap cv_folds at n_samples (CV requires n_splits <= n_samples)
            if train_size < cv_folds:
                cv_folds = max(1, train_size)
                logger.debug(f"Reduced CV folds to {cv_folds} for train_size={train_size}")
            elif train_size < 30:
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

    def _get_global_finetuned_checkpoint_dir(self) -> str:
        """Get checkpoint directory for globally finetuned model from config."""
        # Check models list for checkpoint_dir
        models_list = self.config.get('models', [])
        for m in models_list:
            if m.get('name') == 'tabpfn_global_finetuned' and 'checkpoint_dir' in m:
                return m['checkpoint_dir']
        # Fallback: check top-level config
        return self.config.get('global_finetuned', {}).get('checkpoint_dir', '')

    def _load_data_and_group_by_county(self) -> Tuple[pd.DataFrame, Dict[int, Dict], TestSetResult]:
        """
        Load test set, load data, remap indices, and group by county.

        Returns:
            Tuple of (df, county_data, test_result)
        """
        logger.info("Loading pre-generated test set...")
        test_result = load_test_set_result(self.test_set_dir)

        # Load data
        logger.info("Loading data for pre-generated splits (memory-efficient)...")
        all_indices = np.concatenate([
            test_result.test_indices,
            test_result.train_pool_indices,
        ])
        unique_indices = np.unique(all_indices)
        df = self.data_loader.load_data_by_indices(unique_indices)
        logger.info(f"  Loaded {len(df):,} rows")

        # Optional ratio filter: drop rows with extreme MARKET_TOTAL_VALUE / SALE_AMOUNT
        # Must happen before index mapping so unique_indices stays in sync with df rows.
        # SALE_AMOUNT is log-transformed in the cleaned data, so use exp() for the ratio.
        if self.ratio_filter_enabled:
            if 'MARKET_TOTAL_VALUE' in df.columns:
                df = df.reset_index(drop=True)
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
                unique_indices = unique_indices[keep]
                df = df[keep].reset_index(drop=True)
                logger.info(
                    f"  ratio_filter: {n_before:,} -> {len(df):,} rows "
                    f"(dropped {n_before - len(df):,}; "
                    f"bottom={self.ratio_filter_drop_bottom_percentile}%, "
                    f"top={self.ratio_filter_drop_top_percentile}%)"
                )
            else:
                logger.warning("ratio_filter enabled but MARKET_TOTAL_VALUE not found, skipping")

        # Create index mapping
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

        remapped_test = np.array([index_map[idx] for idx in test_result.test_indices if idx in index_map])
        remapped_train_pool = np.array([index_map[idx] for idx in test_result.train_pool_indices if idx in index_map])

        df = df.reset_index(drop=True)

        # Group by county
        fips_values = df['fips'].values

        county_data = {}
        for fips in test_result.test_counties:
            county_info = test_result.county_info.get(fips, {})
            bucket = county_info.get('size_bucket', 'unknown')

            if bucket not in self.target_buckets:
                continue

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

        return df, county_data, test_result

    def _load_external_neighbor_data(
        self,
        df: pd.DataFrame,
        county_data: Dict[int, Dict],
        test_result: TestSetResult,
        centroids_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        Load data from non-test_v4 counties to extend the neighbor pool.

        For each test county, finds k-nearest neighbors from ALL counties
        (not just test_v4). Loads data for external counties that appear as
        neighbors, applies ratio filter, and appends to the main DataFrame.

        Args:
            df: Current DataFrame (test_v4 data only)
            county_data: Current county_data dict (test_v4 counties only)
            test_result: Test set result (to identify test_v4 counties)
            centroids_df: Centroid coordinates for distance computation

        Returns:
            Tuple of (extended_df, extended_county_data)
        """
        logger.info("=" * 60)
        logger.info("EXTENDING NEIGHBOR POOL BEYOND test_v4")
        logger.info("=" * 60)

        # Step 1: Read fips column to discover all counties
        all_fips_arr = self.data_loader.load_fips_column()

        # Step 2: Build fips → row_indices mapping for non-test_v4 counties
        test_v4_fips = set(test_result.test_counties)
        all_unique_fips = set(np.unique(all_fips_arr))
        external_fips_set = all_unique_fips - test_v4_fips

        logger.info(f"  Total counties in data.parquet: {len(all_unique_fips)}")
        logger.info(f"  test_v4 counties: {len(test_v4_fips)}")
        logger.info(f"  External counties available: {len(external_fips_set)}")

        # Step 3: For each test county, compute k-nearest from ALL counties
        # to find which external counties are actually needed
        needed_external = set()
        external_fips_list = list(external_fips_set)
        for target_fips in county_data.keys():
            neighbors = get_neighbors(
                target_fips=target_fips,
                candidate_fips=external_fips_list,
                centroids_df=centroids_df,
                max_k=self.max_k_neighbors,
                max_distance_miles=self.max_distance_miles,
            )
            needed_external.update(fips for fips, _ in neighbors)

        logger.info(f"  External counties needed as neighbors: {len(needed_external)}")

        if not needed_external:
            logger.info("  No external neighbors needed, skipping")
            return df, county_data

        # Step 4: Build row indices for needed external counties
        needed_external_set = set(needed_external)
        external_row_indices = []
        external_fips_to_rows = {}  # fips → list of positions in external_row_indices

        for row_idx, fips_val in enumerate(all_fips_arr):
            fips_int = int(fips_val)
            if fips_int in needed_external_set:
                if fips_int not in external_fips_to_rows:
                    external_fips_to_rows[fips_int] = []
                external_fips_to_rows[fips_int].append(len(external_row_indices))
                external_row_indices.append(row_idx)

        external_row_indices = np.array(external_row_indices, dtype=np.int64)
        logger.info(f"  Total external rows to load: {len(external_row_indices):,}")

        # Step 5: Load external county data
        external_df = self.data_loader.load_data_by_indices(external_row_indices)
        external_df = external_df.reset_index(drop=True)

        # Step 6: Apply ratio filter to external data
        if self.ratio_filter_enabled and 'MARKET_TOTAL_VALUE' in external_df.columns:
            ratio = external_df['MARKET_TOTAL_VALUE'] / np.exp(external_df[self.target_column])

            if self.ratio_filter_by_sale_year and 'sale_year' in external_df.columns:
                def _pct_rank(g):
                    return g.rank(pct=True) * 100
                ratio_pct = ratio.groupby(external_df['sale_year']).transform(_pct_rank)
            else:
                ratio_pct = ratio.rank(pct=True) * 100

            keep = np.ones(len(external_df), dtype=bool)
            if self.ratio_filter_drop_bottom_percentile > 0:
                keep &= ratio_pct.values >= self.ratio_filter_drop_bottom_percentile
            if self.ratio_filter_drop_top_percentile > 0:
                keep &= ratio_pct.values <= (100 - self.ratio_filter_drop_top_percentile)

            n_before = len(external_df)
            # Update the fips-to-rows mapping to account for dropped rows
            old_to_new = {}
            new_idx = 0
            for old_idx in range(n_before):
                if keep[old_idx]:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1

            # Rebuild fips_to_rows with new indices
            for fips in list(external_fips_to_rows.keys()):
                new_rows = []
                for old_pos in external_fips_to_rows[fips]:
                    if old_pos in old_to_new:
                        new_rows.append(old_to_new[old_pos])
                if new_rows:
                    external_fips_to_rows[fips] = new_rows
                else:
                    del external_fips_to_rows[fips]

            external_df = external_df[keep].reset_index(drop=True)
            logger.info(
                f"  Ratio filter on external data: {n_before:,} -> {len(external_df):,} rows "
                f"(dropped {n_before - len(external_df):,})"
            )

        # Step 7: Append external data to main df
        offset = len(df)
        df = pd.concat([df, external_df], ignore_index=True)
        logger.info(f"  Extended DataFrame: {offset:,} + {len(external_df):,} = {len(df):,} rows")

        # Step 8: Add external counties to county_data
        n_external_added = 0
        for fips, local_indices in external_fips_to_rows.items():
            # Shift local indices by offset to get positions in extended df
            train_pool = np.array(local_indices, dtype=np.int64) + offset
            county_data[fips] = {
                'train_pool_indices': train_pool,
                'test_indices': np.array([], dtype=np.int64),
                'size_bucket': 'external',
                'county_info': {'source': 'external_neighbor'},
                'is_external': True,
            }
            n_external_added += 1

        logger.info(f"  Added {n_external_added} external counties to neighbor pool")
        logger.info(f"  Total counties in pool: {len(county_data)} "
                     f"({len(county_data) - n_external_added} test_v4 + {n_external_added} external)")
        logger.info("=" * 60)

        return df, county_data

    def _build_neighbor_pool(
        self,
        target_fips: int,
        own_train_size: int,
        county_data: Dict[int, Dict],
        centroids_df: pd.DataFrame,
        df: pd.DataFrame,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Build the neighbor portion of the training set for a target county.

        Args:
            target_fips: FIPS of the county being evaluated
            own_train_size: Number of the county's own training samples
            county_data: All counties' data
            centroids_df: Centroid coordinates
            df: Full DataFrame
            rng: Random state for sampling

        Returns:
            Tuple of (neighbor row indices in df, list of usage dicts per neighbor used)
        """
        # Compute neighbor budget
        if self.neighbor_budget_fixed is not None:
            budget = self.neighbor_budget_fixed
        else:
            budget = int(own_train_size * self.neighbor_budget_ratio)

        # Apply total training size cap: own + neighbor <= max_total_training_size
        if self.max_total_training_size is not None:
            budget = min(budget, max(0, self.max_total_training_size - own_train_size))

        if budget <= 0:
            return np.array([], dtype=int), []

        # All other counties that have train pool data
        candidate_fips = [f for f in county_data.keys() if f != target_fips]

        # Optionally restrict to neighbors in the same size bucket as the target
        if self.restrict_neighbors_to_same_size_bucket:
            target_bucket = county_data[target_fips]['size_bucket']
            candidate_fips = [f for f in candidate_fips if county_data[f]['size_bucket'] == target_bucket]

        # Get neighbors sorted by distance
        neighbors = get_neighbors(
            target_fips=target_fips,
            candidate_fips=candidate_fips,
            centroids_df=centroids_df,
            max_k=self.max_k_neighbors,
            max_distance_miles=self.max_distance_miles,
        )

        if not neighbors:
            logger.warning(f"  No neighbors found for county {target_fips}")
            return np.array([], dtype=int), []

        # Sample from neighbors, closest first, until budget is filled
        neighbor_indices = []
        usage = []

        # Diversified sampling: if the closest neighbor alone can fill the
        # entire budget, split it equally among the N nearest instead.
        if (self.diversify_neighbors
                and len(neighbors) >= self.diversify_n):
            first_fips, _ = neighbors[0]
            first_pool_size = len(county_data[first_fips]['train_pool_indices'])
            if first_pool_size >= budget:
                per_neighbor = budget // self.diversify_n
                remainder = budget % self.diversify_n
                for i in range(self.diversify_n):
                    n_fips, dist = neighbors[i]
                    pool = county_data[n_fips]['train_pool_indices']
                    n_take = min(len(pool), per_neighbor + (1 if i < remainder else 0))
                    if n_take > 0:
                        sampled = rng.choice(pool, size=n_take, replace=False)
                        neighbor_indices.extend(sampled)
                        usage.append({
                            'neighbor_fips': n_fips,
                            'distance_miles': dist,
                            'n_samples_taken': n_take,
                            'n_samples_available': len(pool),
                        })
                logger.debug(
                    f"  Diversified sampling for FIPS={target_fips}: "
                    f"split {budget} across {self.diversify_n} neighbors"
                )
                return np.array(neighbor_indices, dtype=int), usage

        # Default: greedy closest-first sampling
        remaining = budget
        for neighbor_fips, dist in neighbors:
            if remaining <= 0:
                break

            pool = county_data[neighbor_fips]['train_pool_indices']
            n_available = len(pool)

            # Apply per-neighbor cap if configured
            n_take = min(n_available, remaining)
            if self.max_samples_per_neighbor is not None:
                n_take = min(n_take, self.max_samples_per_neighbor)

            if n_take > 0:
                sampled = rng.choice(pool, size=n_take, replace=False)
                neighbor_indices.extend(sampled)
                remaining -= n_take
                usage.append({
                    'neighbor_fips': neighbor_fips,
                    'distance_miles': dist,
                    'n_samples_taken': n_take,
                    'n_samples_available': n_available,
                })

        return np.array(neighbor_indices, dtype=int), usage

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
        """Load checkpoint if available."""
        results = []
        completed_keys = set()

        if not self.checkpoint_resume or not self.checkpoint_enabled:
            return results, completed_keys

        checkpoint_csv = results_dir / 'results_checkpoint.csv'
        checkpoint_keys = results_dir / 'completed_keys.pkl'

        if checkpoint_csv.exists() and checkpoint_keys.exists():
            logger.info("Resuming from checkpoint...")
            results_df = pd.read_csv(checkpoint_csv)

            # Deduplicate: keep last entry per (fips, model) to match completed_keys.
            # Duplicates can accumulate from re-runs that loaded old checkpoint results.
            n_before = len(results_df)
            results_df = results_df.drop_duplicates(subset=['fips', 'model'], keep='last')
            n_dropped = n_before - len(results_df)
            if n_dropped > 0:
                logger.warning(f"  Dropped {n_dropped} duplicate checkpoint rows")

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
        pd.DataFrame(results).to_csv(results_dir / 'results_checkpoint.csv', index=False)

        with open(results_dir / 'completed_keys.pkl', 'wb') as f:
            pickle.dump(completed_keys, f)

        logger.debug(f"Checkpoint saved: {len(results)} results")

    def run_experiment(self) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
        """Run the geographic pooling experiment."""
        logger.info("=" * 80)
        logger.info(f"GEO POOLING EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("=" * 80)

        # Load data and group by county
        df, county_data, test_result = self._load_data_and_group_by_county()

        # Load centroids
        centroids_path = Path(self.config['data'].get('project_root', '.')) / self.centroids_csv
        if not centroids_path.exists():
            # Try absolute path
            centroids_path = Path(self.centroids_csv)
        centroids_df = load_centroids(str(centroids_path))

        # Optionally extend neighbor pool with non-test_v4 counties
        if self.extend_neighbor_pool_beyond_testv4:
            df, county_data = self._load_external_neighbor_data(
                df, county_data, test_result, centroids_df
            )

        # Sort counties for deterministic ordering (needed for array job slicing)
        # Only include test_v4 counties (external counties are neighbors only, not targets)
        sorted_fips = sorted(
            fips for fips, cdata in county_data.items()
            if not cdata.get('is_external', False)
        )

        # SLURM array job: select a chunk of counties
        if self.county_index is not None:
            total = len(sorted_fips)
            if self.county_chunk_size is not None:
                chunk_size = self.county_chunk_size
            elif self.n_chunks is not None:
                chunk_size = -(-total // self.n_chunks)  # ceiling division
            else:
                chunk_size = total  # no chunking — process all (shouldn't happen in array mode)
            start = self.county_index * chunk_size
            end = min(start + chunk_size, total)
            sorted_fips = sorted_fips[start:end]
            logger.info(f"Array job chunk {self.county_index}/{self.n_chunks}: "
                        f"counties {start}-{end-1} of {total} "
                        f"({len(sorted_fips)} counties, chunk_size={chunk_size})")

        # Setup output
        results_dir = Path(self.config['output']['results_dir'])
        if self.county_index is not None:
            results_dir = results_dir / f"chunk_{self.county_index}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        all_results, completed_keys = self._load_checkpoint(results_dir)

        # Get enabled models
        enabled_models = self.get_enabled_models()
        seed = self.config.get('experiment', {}).get('random_seed', 42)

        # Log geo pooling config
        logger.info(f"Geo pooling config:")
        logger.info(f"  max_k_neighbors: {self.max_k_neighbors}")
        logger.info(f"  max_distance_miles: {self.max_distance_miles}")
        logger.info(f"  neighbor_budget_ratio: {self.neighbor_budget_ratio}")
        logger.info(f"  neighbor_budget_fixed: {self.neighbor_budget_fixed}")
        logger.info(f"  max_samples_per_neighbor: {self.max_samples_per_neighbor}")
        logger.info(f"  max_total_training_size: {self.max_total_training_size}")
        logger.info(f"  restrict_neighbors_to_same_size_bucket: {self.restrict_neighbors_to_same_size_bucket}")
        logger.info(f"  extend_neighbor_pool_beyond_testv4: {self.extend_neighbor_pool_beyond_testv4}")
        logger.info(f"  diversify_neighbors: {self.diversify_neighbors} (n={self.diversify_n})")
        logger.info(f"Models: {enabled_models}")
        logger.info(f"Counties to process: {len(sorted_fips)}")
        logger.info("=" * 80)

        start_time = time.time()
        counties_since_checkpoint = 0
        all_prediction_frames = []
        all_neighbor_usage = []

        for county_idx, fips in enumerate(sorted_fips):
            cdata = county_data[fips]
            own_pool = cdata['train_pool_indices']

            logger.info(
                f"\n[{county_idx + 1}/{len(sorted_fips)}] "
                f"FIPS={fips} bucket={cdata['size_bucket']} "
                f"own_pool={len(own_pool)} test={len(cdata['test_indices'])}"
            )

            # Build neighbor training pool
            rng = np.random.RandomState(seed)
            neighbor_indices, neighbor_usage = self._build_neighbor_pool(
                target_fips=fips,
                own_train_size=len(own_pool),
                county_data=county_data,
                centroids_df=centroids_df,
                df=df,
                rng=rng,
            )
            for row in neighbor_usage:
                row['target_fips'] = fips
                row['target_own_train_size'] = len(own_pool)
                row['target_test_size'] = len(cdata['test_indices'])
            all_neighbor_usage.extend(neighbor_usage)

            # Combine: all own historical data + neighbor data
            train_indices = np.concatenate([own_pool, neighbor_indices])
            logger.info(f"  Training set: {len(own_pool)} own + {len(neighbor_indices)} neighbor = {len(train_indices)}")

            # Skip if training set is too small for any model to handle
            if len(train_indices) < 2:
                logger.warning(f"  Skipping FIPS={fips}: only {len(train_indices)} training sample(s), need at least 2")
                for model_name in enabled_models:
                    key = (fips, model_name)
                    if key not in completed_keys:
                        result = {
                            'fips': fips,
                            'size_bucket': cdata['size_bucket'],
                            'model': model_name,
                            'own_train_size': len(own_pool),
                            'neighbor_train_size': len(neighbor_indices),
                            'total_train_size': len(train_indices),
                            'test_size': len(cdata['test_indices']),
                            'n_features': 0,
                            'fit_time': 0,
                            'pred_time': 0,
                            'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                            'mape': np.nan, 'mse': np.nan,
                            'status': 'skipped: training set too small',
                        }
                        result = self.metadata.add_to_result(result)
                        all_results.append(result)
                        completed_keys.add(key)
                counties_since_checkpoint += 1
                continue

            # Extract features (raw, before preprocessing)
            X_train_raw, y_train_raw = self._get_features_and_target(df, train_indices)
            X_test_raw, y_test_raw = self._get_features_and_target(df, cdata['test_indices'])

            # Apply Phase 2 preprocessing (fit fresh for each county)
            if self.phase2_config:
                X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
                    X_train=X_train_raw,
                    y_train=y_train_raw,
                    X_test=X_test_raw,
                    y_test=y_test_raw,
                    config=self.phase2_config,
                )
            else:
                X_train, y_train = X_train_raw, y_train_raw
                X_test, y_test = X_test_raw, y_test_raw

            # Train each model
            for model_name in enabled_models:
                key = (fips, model_name)
                if key in completed_keys:
                    logger.info(f"  {model_name}: already done (checkpoint)")
                    continue

                logger.info(f"  Training {model_name}...")

                # Skip finetuning for counties that are too small
                if model_name == 'tabpfn_finetuned' and len(X_train) < self.min_finetune_size:
                    logger.warning(
                        f"  Skipping tabpfn_finetuned for FIPS={fips}: "
                        f"train_size={len(X_train)} < min_finetune_size={self.min_finetune_size}"
                    )
                    result = {
                        'fips': fips, 'size_bucket': cdata['size_bucket'],
                        'model': model_name,
                        'own_train_size': len(own_pool),
                        'neighbor_train_size': len(neighbor_indices),
                        'total_train_size': len(train_indices),
                        'test_size': len(cdata['test_indices']),
                        'n_features': len(X_train.columns),
                        'fit_time': 0, 'pred_time': 0,
                        'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                        'mape': np.nan, 'mse': np.nan,
                        'status': f'skipped: train_size {len(X_train)} < min_finetune_size {self.min_finetune_size}',
                    }
                    result = self.metadata.add_to_result(result)
                    all_results.append(result)
                    completed_keys.add(key)
                    continue

                try:
                    model = self.create_model(model_name, train_size=len(X_train))

                    if model_name == 'tabpfn_global_finetuned':
                        # Pre-finetuned model: no fit(), pass training data as context.
                        # Features are per-county Phase 2 scaled (X_train, X_test), which
                        # matches the per-county x normalization done inside _fit_per_county().
                        fit_start = time.time()
                        fit_time = 0.0  # no fitting needed

                        pred_start = time.time()
                        y_pred = model.predict(X_test, X_context=X_train, y_context=y_train)
                        pred_time = time.time() - pred_start
                    else:
                        fit_start = time.time()
                        model.fit(X_train, y_train)
                        fit_time = time.time() - fit_start

                        pred_start = time.time()
                        y_pred = model.predict(X_test)
                        pred_time = time.time() - pred_start

                    metrics = compute_metrics(y_test.values, y_pred, log_transformed=self.log_transformed)
                    model.cleanup()

                    raw_test = df.iloc[cdata['test_indices']].reset_index(drop=True)
                    raw_test = raw_test.assign(
                        model=model_name,
                        y_true=y_test.values,
                        y_pred=y_pred,
                    )
                    all_prediction_frames.append(raw_test)

                    result = {
                        'fips': fips,
                        'size_bucket': cdata['size_bucket'],
                        'model': model_name,
                        'own_train_size': len(own_pool),
                        'neighbor_train_size': len(neighbor_indices),
                        'total_train_size': len(train_indices),
                        'test_size': len(X_test),
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
                        f"    R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} "
                        f"MAPE={metrics['mape']:.2f} fit={fit_time:.1f}s"
                    )

                except Exception as e:
                    logger.error(f"    Failed: {e}", exc_info=True)
                    result = {
                        'fips': fips,
                        'size_bucket': cdata['size_bucket'],
                        'model': model_name,
                        'own_train_size': len(own_pool),
                        'neighbor_train_size': len(neighbor_indices),
                        'total_train_size': len(train_indices),
                        'test_size': len(cdata['test_indices']),
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
                    # Do NOT add to completed_keys — failures will be retried on resume
                    result = self.metadata.add_to_result(result)
                    all_results.append(result)
                    continue

                result = self.metadata.add_to_result(result)
                all_results.append(result)
                completed_keys.add(key)

            counties_since_checkpoint += 1
            if counties_since_checkpoint >= self.checkpoint_interval:
                self._save_checkpoint(all_results, completed_keys, results_dir)
                counties_since_checkpoint = 0

        # Final save
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()

        if not results_df.empty:
            self.save_results(results_df, results_dir, 'results.csv')

        if all_prediction_frames:
            pred_df = pd.concat(all_prediction_frames, ignore_index=True)
            pred_path = results_dir / 'predictions.parquet'
            pred_df.to_parquet(pred_path, index=False)
            logger.info(f"Predictions saved to {pred_path} ({len(pred_df):,} rows)")

        if all_neighbor_usage:
            usage_df = pd.DataFrame(all_neighbor_usage)[
                ['target_fips', 'target_own_train_size', 'target_test_size',
                 'neighbor_fips', 'distance_miles', 'n_samples_taken', 'n_samples_available']
            ]
            usage_path = results_dir / 'neighbor_usage.parquet'
            usage_df.to_parquet(usage_path, index=False)
            logger.info(f"Neighbor usage saved to {usage_path} ({len(usage_df):,} rows)")

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
