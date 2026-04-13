"""
Tier 1 Diagnostic: Context Sensitivity Test for Globally Finetuned TabPFN

Tests whether the globally finetuned model actually reads the ICL context by:
1. Normal context: county's own train pool as context
2. Shuffled-y context: same X_context but y labels randomly permuted
3. Minimal context: only 2 random samples as context

Also prints raw predictions vs ground truth to check if predictions cluster
near the global prior (y_mean from finetuning).

Runs both zero-shot TabPFN and globally finetuned TabPFN for comparison.

Usage:
    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_context_sensitivity.py

    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_context_sensitivity.py \
        --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/external_15k/
"""

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diag_context_sensitivity")


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


def run_test(model, model_name, X_test, X_context, y_context, y_test,
             log_transformed):
    """Run prediction and compute metrics."""
    from src.evaluation import compute_metrics

    if model_name.startswith("TabPFN (zero-shot)"):
        model.fit(X_context, y_context)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test, X_context=X_context, y_context=y_context)

    metrics = compute_metrics(y_test.values, y_pred, log_transformed=log_transformed)
    return y_pred, metrics


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Context sensitivity diagnostic")
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

    # Build index remap (parquet row idx -> df iloc position)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)}

    # Remap test and train pool indices
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
    # 2. Load models
    # =========================================================================
    logger.info(f"Loading globally finetuned model from {args.checkpoint_dir}...")
    from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel
    ft_model = DirectFineTunedTabPFNModel.load_from_disk(args.checkpoint_dir)
    global_y_mean = ft_model._y_mean
    global_y_std = ft_model._y_std

    logger.info("Loading zero-shot TabPFN...")
    from src.models.tabpfn_wrapper import TabPFNModel
    zs_model = TabPFNModel(device="cuda", version="v2", random_state=args.seed)

    # =========================================================================
    # 3. Run diagnostics per county
    # =========================================================================
    from src.data.preprocessing_utils import apply_phase2_preprocessing

    phase2_config = {
        "winsorize": True,
        "winsorize_percentile": 1,
        "normalize_continuous": True,
        "impute_method": "median",
    }

    print("\n" + "=" * 90)
    print("TIER 1 DIAGNOSTIC: CONTEXT SENSITIVITY TEST")
    print("=" * 90)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Global prior y_mean: {global_y_mean:.4f} (log scale)")
    print(f"Global prior y_std: {global_y_std:.4f}")
    print(f"log_transformed: {log_transformed}")
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

        # Count non-all-NaN float columns in own train pool
        n_cont = int((~X_train_raw.select_dtypes(include="float").isnull().all(axis=0)).sum())

        print("-" * 90)
        print(f"County FIPS={fips} ({bucket}, own_train_size={train_size}, test_size={test_size}, n_cont={n_cont})")
        print(f"  y_train range (log): [{y_train_raw.min():.3f}, {y_train_raw.max():.3f}], "
              f"mean={y_train_raw.mean():.3f}")
        print(f"  y_test  range (log): [{y_test_raw.min():.3f}, {y_test_raw.max():.3f}], "
              f"mean={y_test_raw.mean():.3f}")
        print(f"  Global prior (y_mean): {global_y_mean:.3f}")
        print()

        # Prepare 3 context variants
        contexts = {}

        # Normal context
        X_tr, y_tr, X_te, y_te = apply_phase2_preprocessing(
            X_train=X_train_raw.copy(), y_train=y_train_raw.copy(),
            X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
            config=phase2_config,
        )
        contexts["Normal"] = (X_tr, y_tr, X_te, y_te)

        # Shuffled-y context
        y_shuffled = y_train_raw.sample(frac=1.0, random_state=args.seed)
        y_shuffled.index = y_train_raw.index  # keep original index alignment
        X_tr_s, y_tr_s, X_te_s, y_te_s = apply_phase2_preprocessing(
            X_train=X_train_raw.copy(), y_train=y_shuffled.copy(),
            X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
            config=phase2_config,
        )
        contexts["Shuffled-y"] = (X_tr_s, y_tr_s, X_te_s, y_te_s)

        # Minimal context (2 random samples)
        if train_size > 2:
            mini_idx = rng.choice(len(X_train_raw), size=2, replace=False)
            X_mini = X_train_raw.iloc[mini_idx]
            y_mini = y_train_raw.iloc[mini_idx]
        else:
            X_mini = X_train_raw.copy()
            y_mini = y_train_raw.copy()
        X_tr_m, y_tr_m, X_te_m, y_te_m = apply_phase2_preprocessing(
            X_train=X_mini.copy(), y_train=y_mini.copy(),
            X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
            config=phase2_config,
        )
        contexts["Minimal (2)"] = (X_tr_m, y_tr_m, X_te_m, y_te_m)

        # Run all tests
        header = f"  {'Model':<35} | {'Context':<15} | {'MAPE':>7} | {'R²':>8} | {'MAE':>7} | {'RMSE':>7}"
        sep = f"  {'-'*35}-+-{'-'*15}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}"
        print(header)
        print(sep)

        ft_normal_preds = None

        for ctx_name, (X_ctx, y_ctx, X_te_ctx, y_te_ctx) in contexts.items():
            for model, model_name in [
                (zs_model, "TabPFN (zero-shot)"),
                (ft_model, "TabPFN (globally finetuned)"),
            ]:
                try:
                    y_pred, metrics = run_test(
                        model, model_name, X_te_ctx, X_ctx, y_ctx, y_te_ctx,
                        log_transformed=log_transformed,
                    )
                    mape = metrics.get("mape", float("nan"))
                    r2 = metrics.get("r2", float("nan"))
                    mae = metrics.get("mae", float("nan"))
                    rmse = metrics.get("rmse", float("nan"))
                    print(f"  {model_name:<35} | {ctx_name:<15} | {mape:>7.1f} | {r2:>8.4f} | {mae:>7.0f} | {rmse:>7.0f}")
                    rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                                 "test_size": test_size, "n_cont": n_cont, "model": model_name,
                                 "context": ctx_name, "mape": mape, "r2": r2, "mae": mae, "rmse": rmse})

                    # Save finetuned normal predictions for inspection
                    if model_name.startswith("TabPFN (globally") and ctx_name == "Normal":
                        ft_normal_preds = y_pred
                except Exception as e:
                    print(f"  {model_name:<35} | {ctx_name:<15} | ERROR: {e}")
                    rows.append({"fips": fips, "bucket": bucket, "own_train_size": train_size,
                                 "test_size": test_size, "model": model_name, "context": ctx_name,
                                 "mape": float("nan"), "r2": float("nan"), "mae": float("nan"), "rmse": float("nan")})

        # Predictions inspection
        if ft_normal_preds is not None:
            _, _, X_te_norm, y_te_norm = contexts["Normal"]
            n_show = min(10, len(ft_normal_preds))
            print()
            print(f"  Predictions (globally finetuned, normal context, first {n_show}):")
            print(f"    y_pred (log): [{', '.join(f'{v:.3f}' for v in ft_normal_preds[:n_show])}]")
            print(f"    y_true (log): [{', '.join(f'{v:.3f}' for v in y_te_norm.values[:n_show])}]")
            if log_transformed:
                print(f"    y_pred (exp): [{', '.join(f'{np.exp(v):,.0f}' for v in ft_normal_preds[:n_show])}]")
                print(f"    y_true (exp): [{', '.join(f'{np.exp(v):,.0f}' for v in y_te_norm.values[:n_show])}]")

        print()

    # Save per-FIPS results to CSV
    import os
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    csv_path = f"logs/debugging/finetuning/diag_context_sensitivity_{job_id}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Per-FIPS results saved to {csv_path}")

    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)
    print()
    print("Interpretation guide:")
    print("  - If shuffled-y MAPE ≈ normal MAPE for finetuned → model ignores context (ICL broken)")
    print("  - If shuffled-y MAPE >> normal MAPE for finetuned → model reads context, problem elsewhere")
    print("  - If finetuned predictions cluster near global prior (y_mean) → prior-collapse failure")
    print("  - Compare zero-shot vs finetuned: zero-shot should degrade with shuffled-y")


if __name__ == "__main__":
    main()
