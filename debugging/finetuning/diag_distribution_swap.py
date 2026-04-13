"""
Tier 2 Diagnostic: Distribution Swap Test for Globally Finetuned TabPFN

Tests whether the model's ICL works when given context from the finetuning
training distribution (mixed counties) vs the county's own data.

If performance recovers with finetuning-distribution context, the model's ICL
is intact but distribution-specific — pointing toward LoRA or diverse training.
If performance is still terrible, the ICL mechanism is broadly broken.

Usage:
    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_distribution_swap.py

    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_distribution_swap.py \
        --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/external_15k/ \
        --n_swap_samples 3000
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diag_distribution_swap")


# Same exclude columns as geo_pooling.py
EXCLUDE_COLUMNS = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]


def get_features_and_target(df, indices, target_column):
    """Extract X, y from DataFrame at given iloc positions."""
    subset = df.iloc[indices]
    y = subset[target_column]
    X = subset.drop(
        columns=[target_column] + [c for c in EXCLUDE_COLUMNS if c in subset.columns]
    )
    return X, y


def select_counties(test_result, county_data, n_tiny, n_medium):
    """Auto-select a few tiny and medium counties that have enough data."""
    selected = []
    size_buckets = test_result.size_buckets

    for bucket, n_want in [("tiny", n_tiny), ("medium", n_medium)]:
        fips_list = size_buckets.get(bucket, [])
        count = 0
        for fips in fips_list:
            if fips not in county_data:
                continue
            cdata = county_data[fips]
            train_size = len(cdata["train_pool_indices"])
            test_size = len(cdata["test_indices"])
            if train_size >= 5 and test_size >= 5:
                selected.append((fips, bucket, train_size, test_size))
                count += 1
                if count >= n_want:
                    break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Tier 2: Distribution swap diagnostic")
    parser.add_argument(
        "--checkpoint_dir",
        default="/nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/internal_15k/",
    )
    parser.add_argument(
        "--data_path",
        default="/nlp/scr/salilg/property_tax/preprocessed/v2_no_onehot/",
    )
    parser.add_argument(
        "--test_set_dir",
        default="/nlp/scr/salilg/property_tax/preprocessed/v2_no_onehot/test_v4/",
    )
    parser.add_argument("--n_tiny", type=int, default=3)
    parser.add_argument("--n_medium", type=int, default=3)
    parser.add_argument("--n_swap_samples", type=int, default=2000,
                        help="Number of samples to draw from finetuning distribution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_column = "SALE_AMOUNT"
    rng = np.random.RandomState(args.seed)

    # =========================================================================
    # 1. Load data
    # =========================================================================
    logger.info("Loading test split...")
    from src.data.split_strategies import load_test_set_result
    test_result = load_test_set_result(args.test_set_dir)

    logger.info("Loading data...")
    from src.data.loading import CleanedDataLoader
    loader = CleanedDataLoader(
        cleaned_data_path=args.data_path,
        target_column=target_column,
    )

    all_indices = np.unique(np.concatenate([
        test_result.test_indices, test_result.train_pool_indices
    ]))
    df = loader.load_data_by_indices(all_indices)
    log_transformed = loader.is_target_log_transformed()

    # Build index remap
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)}

    remapped_test = np.array([
        index_map[idx] for idx in test_result.test_indices if idx in index_map
    ])
    remapped_train_pool = np.array([
        index_map[idx] for idx in test_result.train_pool_indices if idx in index_map
    ])

    # Group by county
    fips_col = df["fips"].values
    county_data = {}
    for fips in test_result.test_counties:
        fips_int = int(fips)
        train_mask = np.isin(fips_col[remapped_train_pool], [fips_int])
        test_mask = np.isin(fips_col[remapped_test], [fips_int])
        county_train = remapped_train_pool[train_mask]
        county_test = remapped_test[test_mask]
        if len(county_train) > 0 and len(county_test) > 0:
            county_data[fips_int] = {
                "train_pool_indices": county_train,
                "test_indices": county_test,
            }

    # Select counties
    selected = select_counties(test_result, county_data, args.n_tiny, args.n_medium)
    logger.info(f"Selected {len(selected)} counties: {[(f, b) for f, b, _, _ in selected]}")

    # =========================================================================
    # 2. Prepare finetuning-distribution context
    # =========================================================================
    # For the internal variant, the finetuning used the train_pool_indices
    # (the 80% temporal train pool across all test_v4 counties).
    # We sample n_swap_samples from these as the "finetuning distribution" context.
    n_available = len(remapped_train_pool)
    n_take = min(args.n_swap_samples, n_available)
    swap_indices = rng.choice(remapped_train_pool, size=n_take, replace=False)
    logger.info(f"Finetuning-distribution context: {n_take} samples from train pool "
                f"({n_available} available)")

    X_swap_raw, y_swap_raw = get_features_and_target(df, swap_indices, target_column)
    logger.info(f"  y_swap range (log): [{y_swap_raw.min():.3f}, {y_swap_raw.max():.3f}], "
                f"mean={y_swap_raw.mean():.3f}")

    # =========================================================================
    # 3. Load models
    # =========================================================================
    logger.info(f"Loading globally finetuned model from {args.checkpoint_dir}...")
    from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel
    ft_model = DirectFineTunedTabPFNModel.load_from_disk(args.checkpoint_dir)
    global_y_mean = ft_model._y_mean

    logger.info("Loading zero-shot TabPFN...")
    from src.models.tabpfn_wrapper import TabPFNModel
    zs_model = TabPFNModel(device="cuda", version="v2", random_state=args.seed)

    # =========================================================================
    # 4. Run diagnostics per county
    # =========================================================================
    from src.data.preprocessing_utils import apply_phase2_preprocessing
    from src.evaluation import compute_metrics

    phase2_config = {
        "winsorize": True,
        "winsorize_percentile": 1,
        "normalize_continuous": True,
        "impute_method": "median",
    }

    print("\n" + "=" * 90)
    print("TIER 2 DIAGNOSTIC: DISTRIBUTION SWAP TEST")
    print("=" * 90)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Global prior y_mean: {global_y_mean:.4f} (log scale)")
    print(f"Swap context: {n_take} samples from train pool")
    print()

    rows = []  # accumulate per-FIPS results for CSV

    for fips, bucket, train_size, test_size in selected:
        cdata = county_data[fips]
        X_train_raw, y_train_raw = get_features_and_target(
            df, cdata["train_pool_indices"], target_column
        )
        X_test_raw, y_test_raw = get_features_and_target(
            df, cdata["test_indices"], target_column
        )

        print("-" * 90)
        print(f"County FIPS={fips} ({bucket}, own_train_size={train_size}, test_size={test_size})")
        print(f"  y_train mean (log): {y_train_raw.mean():.3f}, "
              f"y_swap mean (log): {y_swap_raw.mean():.3f}, "
              f"global prior: {global_y_mean:.3f}")
        print()

        header = f"  {'Model':<35} | {'Context source':<25} | {'MAPE':>7} | {'R²':>8} | {'MAE':>7}"
        sep = f"  {'-'*35}-+-{'-'*25}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}"
        print(header)
        print(sep)

        # Test A: Zero-shot with county's own context
        try:
            X_tr_a, y_tr_a, X_te_a, y_te_a = apply_phase2_preprocessing(
                X_train=X_train_raw.copy(), y_train=y_train_raw.copy(),
                X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
                config=phase2_config,
            )
            zs_model.fit(X_tr_a, y_tr_a)
            y_pred_a = zs_model.predict(X_te_a)
            m = compute_metrics(y_te_a.values, y_pred_a, log_transformed=log_transformed)
            print(f"  {'TabPFN (zero-shot)':<35} | {'County own (' + str(train_size) + ')':<25} | "
                  f"{m['mape']:>7.1f} | {m['r2']:>8.4f} | {m['mae']:>7.0f}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (zero-shot)",
                         "context_source": "county_own", "n_context_samples": train_size,
                         "mape": m["mape"], "r2": m["r2"], "mae": m["mae"]})
        except Exception as e:
            print(f"  {'TabPFN (zero-shot)':<35} | {'County own':<25} | ERROR: {e}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (zero-shot)",
                         "context_source": "county_own", "n_context_samples": train_size,
                         "mape": float("nan"), "r2": float("nan"), "mae": float("nan")})

        # Test B: Finetuned with county's own context
        try:
            X_tr_b, y_tr_b, X_te_b, y_te_b = apply_phase2_preprocessing(
                X_train=X_train_raw.copy(), y_train=y_train_raw.copy(),
                X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
                config=phase2_config,
            )
            y_pred_b = ft_model.predict(X_te_b, X_context=X_tr_b, y_context=y_tr_b)
            m = compute_metrics(y_te_b.values, y_pred_b, log_transformed=log_transformed)
            print(f"  {'TabPFN (globally finetuned)':<35} | {'County own (' + str(train_size) + ')':<25} | "
                  f"{m['mape']:>7.1f} | {m['r2']:>8.4f} | {m['mae']:>7.0f}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (globally finetuned)",
                         "context_source": "county_own", "n_context_samples": train_size,
                         "mape": m["mape"], "r2": m["r2"], "mae": m["mae"]})
        except Exception as e:
            print(f"  {'TabPFN (globally finetuned)':<35} | {'County own':<25} | ERROR: {e}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (globally finetuned)",
                         "context_source": "county_own", "n_context_samples": train_size,
                         "mape": float("nan"), "r2": float("nan"), "mae": float("nan")})

        # Test C: Finetuned with finetuning-distribution context
        # Phase 2 preprocessing is fit on the swap context, applied to county test
        try:
            X_tr_c, y_tr_c, X_te_c, y_te_c = apply_phase2_preprocessing(
                X_train=X_swap_raw.copy(), y_train=y_swap_raw.copy(),
                X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
                config=phase2_config,
            )
            y_pred_c = ft_model.predict(X_te_c, X_context=X_tr_c, y_context=y_tr_c)
            m = compute_metrics(y_te_c.values, y_pred_c, log_transformed=log_transformed)
            print(f"  {'TabPFN (globally finetuned)':<35} | {'Finetuning dist (' + str(n_take) + ')':<25} | "
                  f"{m['mape']:>7.1f} | {m['r2']:>8.4f} | {m['mae']:>7.0f}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (globally finetuned)",
                         "context_source": "finetuning_dist", "n_context_samples": n_take,
                         "mape": m["mape"], "r2": m["r2"], "mae": m["mae"]})
        except Exception as e:
            print(f"  {'TabPFN (globally finetuned)':<35} | {'Finetuning dist':<25} | ERROR: {e}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (globally finetuned)",
                         "context_source": "finetuning_dist", "n_context_samples": n_take,
                         "mape": float("nan"), "r2": float("nan"), "mae": float("nan")})

        # Test D: Zero-shot with finetuning-distribution context (for comparison)
        try:
            X_tr_d, y_tr_d, X_te_d, y_te_d = apply_phase2_preprocessing(
                X_train=X_swap_raw.copy(), y_train=y_swap_raw.copy(),
                X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
                config=phase2_config,
            )
            zs_model.fit(X_tr_d, y_tr_d)
            y_pred_d = zs_model.predict(X_te_d)
            m = compute_metrics(y_te_d.values, y_pred_d, log_transformed=log_transformed)
            print(f"  {'TabPFN (zero-shot)':<35} | {'Finetuning dist (' + str(n_take) + ')':<25} | "
                  f"{m['mape']:>7.1f} | {m['r2']:>8.4f} | {m['mae']:>7.0f}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (zero-shot)",
                         "context_source": "finetuning_dist", "n_context_samples": n_take,
                         "mape": m["mape"], "r2": m["r2"], "mae": m["mae"]})
        except Exception as e:
            print(f"  {'TabPFN (zero-shot)':<35} | {'Finetuning dist':<25} | ERROR: {e}")
            rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                         "test_size": test_size, "model": "TabPFN (zero-shot)",
                         "context_source": "finetuning_dist", "n_context_samples": n_take,
                         "mape": float("nan"), "r2": float("nan"), "mae": float("nan")})

        print()

    # Save per-FIPS results to CSV
    import os
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    csv_path = f"logs/debugging/finetuning/diag_distribution_swap_{job_id}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Per-FIPS results saved to {csv_path}")

    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)
    print()
    print("Interpretation guide:")
    print("  - If finetuned + finetuning dist >> finetuned + county own:")
    print("      ICL works but is distribution-specific → LoRA / diverse training")
    print("  - If finetuned + finetuning dist ≈ finetuned + county own (both bad):")
    print("      ICL mechanism is broadly broken → deeper issue")
    print("  - Compare zero-shot + county own vs zero-shot + finetuning dist:")
    print("      Shows whether county-specific context is better than mixed context")


if __name__ == "__main__":
    main()
