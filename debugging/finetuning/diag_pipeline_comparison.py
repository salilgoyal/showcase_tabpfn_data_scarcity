"""
Diagnostic: Pipeline Comparison — Finetuned Weights vs Preprocessing Pipeline

Tests whether the ~15% MAPE degradation of globally finetuned TabPFN comes from
the finetuned weights or the different preprocessing pipeline.

Three models are compared on the same per-county data:
  (A) Zero-shot TabPFN via TabPFNRegressor        — standard pipeline
  (B) Zero-shot TabPFN via DirectFineTunedTabPFNModel — finetuning pipeline, ORIGINAL weights
  (C) Finetuned TabPFN via DirectFineTunedTabPFNModel — finetuning pipeline, FINETUNED weights

Interpretation:
  If (A) ≈ (B) and (A) < (C): pipeline is fine → finetuning hurts the weights
  If (A) < (B) ≈ (C):         pipeline itself degrades performance
  If (A) < (B) < (C):         both pipeline and finetuning contribute

Uses each county's own train pool (not full geo-pooled data) for simplicity.
Only counties with own_train_size >= --min_train_size are tested.

Usage:
    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_pipeline_comparison.py

    # With custom args:
    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_pipeline_comparison.py \
        --n_counties 30 --min_train_size 50
"""

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diag_pipeline_comparison")


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


def select_counties(county_data, fips_stats_df, n_counties, min_train_size, min_test_size):
    """Select counties spanning the n_cont range, with enough data."""
    eligible = fips_stats_df[
        (fips_stats_df["own_train_size"] >= min_train_size)
        & (fips_stats_df["test_size"] >= min_test_size)
    ].copy()

    if len(eligible) <= n_counties:
        return eligible

    # Stratified sample across n_cont range
    eligible = eligible.sort_values("n_cont")
    indices = np.linspace(0, len(eligible) - 1, n_counties, dtype=int)
    return eligible.iloc[indices]


def build_zs_via_ft_pipeline(checkpoint_dir, device):
    """Build a zero-shot (original weights) model using the finetuning pipeline's metadata.

    Loads the finetuned checkpoint to get metadata (continuous_cols, y_mean, y_std,
    cat_cardinalities, _pred_transform), then replaces the model weights with the
    original pretrained checkpoint weights.
    """
    from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel, TabPFN2

    # Load finetuned model (gets all metadata + finetuned weights)
    model = DirectFineTunedTabPFNModel.load_from_disk(checkpoint_dir, device=device)

    # Replace finetuned weights with original pretrained weights
    checkpoint_path = model._find_checkpoint()
    n_num = len([c for c in model._continuous_cols if c in model._all_columns])
    cat_cards = model._cat_cardinalities or []
    device_obj = next(model.model.parameters()).device

    logger.info(f"  Replacing finetuned weights with original checkpoint: {checkpoint_path}")
    model.model = TabPFN2(
        n_num_features=n_num,
        cat_cardinalities=cat_cards,
        n_classes=5000,
        is_regression=True,
        checkpoint_path=checkpoint_path,
    ).to(device_obj)
    model.model.eval()

    # Note: _pred_transform and _y_mean/_y_std are kept from the finetuned checkpoint.
    # This is intentional: we want to test whether the ORIGINAL weights produce
    # reasonable predictions when processed through the finetuning pipeline's
    # standardization (global y_mean/y_std) and output transform.

    return model


def main():
    parser = argparse.ArgumentParser(description="Pipeline comparison diagnostic")
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
    parser.add_argument("--n_counties", type=int, default=30)
    parser.add_argument("--min_train_size", type=int, default=50)
    parser.add_argument("--min_test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_column = "SALE_AMOUNT"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    fips_stats_records = []
    float_cols = [
        c for c in df.columns
        if c not in set(EXCLUDE_COLUMNS + [target_column])
        and pd.api.types.is_float_dtype(df[c])
    ]

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
            X_own = df.iloc[county_train][float_cols]
            n_cont = int((~X_own.isnull().all(axis=0)).sum())
            fips_stats_records.append({
                "fips": fips_int,
                "own_train_size": len(county_train),
                "test_size": len(county_test),
                "n_cont": n_cont,
            })

    fips_stats_df = pd.DataFrame(fips_stats_records)

    # Select counties
    selected = select_counties(
        county_data, fips_stats_df,
        args.n_counties, args.min_train_size, args.min_test_size,
    )
    logger.info(f"Selected {len(selected)} counties (n_cont range: "
                f"{selected['n_cont'].min()}-{selected['n_cont'].max()}, "
                f"own_train_size range: {selected['own_train_size'].min()}-{selected['own_train_size'].max()})")

    # =========================================================================
    # 2. Load models
    # =========================================================================
    logger.info(f"Loading finetuned model from {args.checkpoint_dir}...")
    from src.models.tabpfn_finetuning_v2 import DirectFineTunedTabPFNModel
    ft_model = DirectFineTunedTabPFNModel.load_from_disk(args.checkpoint_dir, device=device)
    global_y_mean = ft_model._y_mean
    global_y_std = ft_model._y_std

    logger.info("Building zero-shot model via finetuning pipeline (original weights)...")
    zs_ft_model = build_zs_via_ft_pipeline(args.checkpoint_dir, device=device)

    logger.info("Loading zero-shot TabPFN via TabPFNRegressor...")
    from src.models.tabpfn_wrapper import TabPFNModel
    zs_model = TabPFNModel(device=device, version="v2", random_state=args.seed)

    # =========================================================================
    # 3. Run per-county comparison
    # =========================================================================
    from src.data.preprocessing_utils import apply_phase2_preprocessing
    from src.evaluation import compute_metrics

    phase2_config = {
        "winsorize": True,
        "winsorize_percentile": 1,
        "normalize_continuous": True,
        "impute_method": "median",
    }

    print("\n" + "=" * 100)
    print("PIPELINE COMPARISON DIAGNOSTIC")
    print("=" * 100)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Global prior y_mean: {global_y_mean:.4f}, y_std: {global_y_std:.4f}")
    print(f"Counties: {len(selected)}, min_train_size: {args.min_train_size}")
    print()

    models_config = [
        ("(A) ZS TabPFNRegressor", "zs_regressor"),
        ("(B) ZS FT-pipeline",     "zs_ft_pipeline"),
        ("(C) FT FT-pipeline",     "ft_pipeline"),
    ]

    rows = []

    for idx, (_, county_row) in enumerate(selected.iterrows()):
        fips = int(county_row["fips"])
        cdata = county_data[fips]

        X_train_raw, y_train_raw = get_features_and_target(
            df, cdata["train_pool_indices"], target_column
        )
        X_test_raw, y_test_raw = get_features_and_target(
            df, cdata["test_indices"], target_column
        )

        n_cont = int(county_row["n_cont"])
        own_train_size = len(cdata["train_pool_indices"])
        test_size = len(cdata["test_indices"])

        # Phase 2 preprocessing (fit on train, apply to both)
        X_train, y_train, X_test, y_test = apply_phase2_preprocessing(
            X_train=X_train_raw.copy(), y_train=y_train_raw.copy(),
            X_test=X_test_raw.copy(), y_test=y_test_raw.copy(),
            config=phase2_config,
        )

        county_y_mean = float(y_train.mean())
        county_y_std = float(y_train.std())

        print(f"[{idx+1}/{len(selected)}] FIPS={fips}  n_cont={n_cont}  "
              f"train={own_train_size}  test={test_size}  "
              f"county_y_mean={county_y_mean:.3f} (global={global_y_mean:.3f})")

        for model_label, model_tag in models_config:
            t0 = time.time()
            try:
                if model_tag == "zs_regressor":
                    zs_model.fit(X_train, y_train)
                    y_pred = zs_model.predict(X_test)
                elif model_tag == "zs_ft_pipeline":
                    y_pred = zs_ft_model.predict(X_test, X_context=X_train, y_context=y_train)
                elif model_tag == "ft_pipeline":
                    y_pred = ft_model.predict(X_test, X_context=X_train, y_context=y_train)
                else:
                    raise ValueError(f"Unknown model tag: {model_tag}")

                elapsed = time.time() - t0
                metrics = compute_metrics(y_test.values, y_pred, log_transformed=log_transformed)

                print(f"  {model_label:28s}  MAPE={metrics['mape']:7.1f}  "
                      f"R2={metrics['r2']:.3f}  MAE={metrics['mae']:.0f}  ({elapsed:.1f}s)")

                rows.append({
                    "fips": fips,
                    "n_cont": n_cont,
                    "own_train_size": own_train_size,
                    "test_size": test_size,
                    "county_y_mean": county_y_mean,
                    "county_y_std": county_y_std,
                    "model": model_tag,
                    "model_label": model_label,
                    "mape": metrics["mape"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "time": elapsed,
                })

            except Exception as e:
                elapsed = time.time() - t0
                logger.error(f"  {model_label}: FAILED — {e}")
                rows.append({
                    "fips": fips,
                    "n_cont": n_cont,
                    "own_train_size": own_train_size,
                    "test_size": test_size,
                    "county_y_mean": county_y_mean,
                    "county_y_std": county_y_std,
                    "model": model_tag,
                    "model_label": model_label,
                    "mape": np.nan,
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "time": elapsed,
                })

    # =========================================================================
    # 4. Save results and print summary
    # =========================================================================
    results = pd.DataFrame(rows)

    # Save CSV
    job_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    import os
    slurm_job_id = os.environ.get("SLURM_JOB_ID", job_id)
    csv_path = f"logs/debugging/finetuning/diag_pipeline_comparison_{slurm_job_id}.csv"
    results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Median MAPE by model")
    print("=" * 100)
    summary = results.groupby("model_label")["mape"].agg(["median", "mean", "std", "count"])
    print(summary.to_string())

    # Pairwise comparison
    print("\n" + "-" * 100)
    print("INTERPRETATION GUIDE")
    print("-" * 100)

    pivot = results.pivot_table(index="fips", columns="model", values="mape")
    if "zs_regressor" in pivot.columns and "zs_ft_pipeline" in pivot.columns:
        ratio_b_vs_a = (pivot["zs_ft_pipeline"] / pivot["zs_regressor"]).median()
        print(f"\n(B) vs (A) — pipeline effect:    median ratio = {ratio_b_vs_a:.3f}")
        if abs(ratio_b_vs_a - 1) < 0.05:
            print("  → Pipeline difference is SMALL (< 5%). The preprocessing is not the issue.")
        else:
            print(f"  → Pipeline difference is SIGNIFICANT ({(ratio_b_vs_a-1)*100:+.1f}%). "
                  "Investigate feature processing differences.")

    if "zs_ft_pipeline" in pivot.columns and "ft_pipeline" in pivot.columns:
        ratio_c_vs_b = (pivot["ft_pipeline"] / pivot["zs_ft_pipeline"]).median()
        print(f"\n(C) vs (B) — weight change effect: median ratio = {ratio_c_vs_b:.3f}")
        if abs(ratio_c_vs_b - 1) < 0.05:
            print("  → Finetuned weights are NOT the issue. Look elsewhere.")
        else:
            print(f"  → Finetuned weights cause a {(ratio_c_vs_b-1)*100:+.1f}% change. "
                  "The finetuning itself is the problem.")

    if "zs_regressor" in pivot.columns and "ft_pipeline" in pivot.columns:
        ratio_c_vs_a = (pivot["ft_pipeline"] / pivot["zs_regressor"]).median()
        print(f"\n(C) vs (A) — total effect:        median ratio = {ratio_c_vs_a:.3f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
