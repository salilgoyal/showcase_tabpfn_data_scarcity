"""
Training Spike Diagnostic for Per-County TabPFN Finetuning

Instruments the training loop to capture detailed per-step information during
loss spikes. Also runs zero-shot per-county evaluation on ALL counties to
determine whether spikes are structural (data/architecture) or training-induced.

Output files (saved to --output_dir):
  1. spike_diagnostics.jsonl — one JSON record per spike step
  2. zeroshot_per_county.json — per-county zero-shot loss at initialization
  3. history.json — full epoch-level training history
  4. model.pt, metadata.json, transforms.pkl — standard checkpoint files

Usage:
    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_training_spikes.py \
        --lora_rank 8 --learning_rate 1e-4 --epoch_size 100 --max_epochs 50

    sbatch debugging/finetuning/run_diagnostic.sh \
        debugging/finetuning/diag_training_spikes.py \
        --spike_threshold 5 --lora_rank 4 --learning_rate 5e-5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diag_training_spikes")


# Same exclude columns as global_finetuning.py and other diagnostics
EXCLUDE_COLUMNS = [
    "fips", "CLIP", "sale_date",
    "Unnamed: 0", "ASSESSED_YEAR", "CENSUS_ID", "PREVIOUS_CLIP",
    "OWNER_TRANSFER_COMPOSITE_TRANSACTION_ID", "address",
    "TOTAL_TAX_AMOUNT", "NET_TAX_AMOUNT", "TAX_RATE_AREA_CODE",
    "CALCULATED_TOTAL_VALUE_SOURCE_CODE", "tract", "block_group",
    "tract_id", "block_group_id", "MULTI_OR_SPLIT_PARCEL_CODE", "meta_sfh",
    "CALCULATED_TOTAL_VALUE",
]


def main():
    parser = argparse.ArgumentParser(
        description="Training spike diagnostic for per-county TabPFN finetuning"
    )

    # Data args (same defaults as other diagnostics)
    parser.add_argument(
        "--data_path",
        default="/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/",
    )
    parser.add_argument(
        "--test_set_dir",
        default="/scratch/users/salilg/property_tax/preprocessed/v2_no_onehot/test_v4_rand_s0/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=999999,
                        help="Max training samples (999999 = use all)")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--epoch_size", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: same as lora_rank)")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_lr_warmup_epochs", type=int, default=0)

    # Spike diagnostic args
    parser.add_argument("--spike_threshold", type=float, default=10.0,
                        help="Log diagnostics when step loss exceeds this")

    # Ratio filter (same as sweep configs)
    parser.add_argument("--ratio_filter", action="store_true", default=True)
    parser.add_argument("--no_ratio_filter", action="store_false", dest="ratio_filter")
    parser.add_argument("--ratio_drop_bottom_pct", type=float, default=5.0)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = float(args.lora_rank)

    # Determine output dir
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    if args.output_dir is None:
        tag = (f"lora{args.lora_rank}_lr{args.learning_rate}_"
               f"ep{args.epoch_size}_thresh{args.spike_threshold}")
        output_dir = Path(f"logs/debugging/finetuning/spike_diag_{tag}_{job_id}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TRAINING SPIKE DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Config: lora_rank={args.lora_rank}, lr={args.learning_rate}, "
                f"epoch_size={args.epoch_size}, max_epochs={args.max_epochs}")
    logger.info(f"  Spike threshold: {args.spike_threshold}")

    # =========================================================================
    # 1. Load data (same pipeline as global_finetuning.py)
    # =========================================================================
    logger.info("Loading test split...")
    from src.data.split_strategies import load_test_set_result
    from src.data.loading import CleanedDataLoader
    from src.data.preprocessing_utils import Phase2Preprocessor

    target_column = "SALE_AMOUNT"
    test_result = load_test_set_result(args.test_set_dir)

    loader = CleanedDataLoader(
        cleaned_data_path=args.data_path,
        target_column=target_column,
    )

    # Internal variant: use train pool
    indices = test_result.train_pool_indices
    rng = np.random.RandomState(args.seed)
    if len(indices) > args.n_samples:
        indices = rng.choice(indices, size=args.n_samples, replace=False)
    indices = np.sort(indices)

    df = loader.load_data_by_indices(indices)

    # Apply ratio filter (same as sweep configs)
    if args.ratio_filter and 'MARKET_TOTAL_VALUE' in df.columns:
        ratio = df['MARKET_TOTAL_VALUE'] / np.exp(df[target_column])
        if 'sale_year' in df.columns:
            ratio_pct = ratio.groupby(df['sale_year']).transform(
                lambda g: g.rank(pct=True) * 100
            )
        else:
            ratio_pct = ratio.rank(pct=True) * 100

        keep = ratio_pct.values >= args.ratio_drop_bottom_pct
        n_before = len(df)
        df = df[keep].reset_index(drop=True)
        logger.info(f"  Ratio filter: {n_before:,} -> {len(df):,} rows")

    # Extract county IDs before dropping fips
    county_ids = df['fips'].copy()

    # Extract features and target
    y = df[target_column]
    X = df.drop(
        columns=[target_column] + [c for c in EXCLUDE_COLUMNS if c in df.columns]
    )

    # Phase 2 preprocessing (skip StandardScaler for per_county mode)
    phase2_config = {
        "winsorize": True,
        "winsorize_percentile": 1,
        "normalize_continuous": False,  # per_county mode normalizes per step
        "impute_method": "median",
    }
    phase2 = Phase2Preprocessor(phase2_config)
    phase2.fit(X, y)
    X = phase2.transform(X)
    y = phase2.transform_target(y)

    logger.info(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features, "
                f"{county_ids.nunique()} counties")

    # =========================================================================
    # 2. Build config with spike diagnostics enabled
    # =========================================================================
    from src.models.tabpfn_finetuning_v2 import (
        DirectFineTunedTabPFNModel, FinetuningConfigV2
    )

    cfg = FinetuningConfigV2(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        epoch_size=args.epoch_size,
        gradient_clip=args.gradient_clip,
        use_amp=True,
        finetune_mode="lora",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        n_lr_warmup_epochs=args.n_lr_warmup_epochs,
        softmax_temperature=0.9,
        val_fraction=0.2,
        eval_batch_size=4096,
        device="cuda",
        random_state=args.seed,
        training_mode="per_county",
        min_county_size=5,
        context_fraction_range=(0.3, 0.7),
        spike_diagnostics=True,
        spike_threshold=args.spike_threshold,
    )

    # =========================================================================
    # 3. Train with spike diagnostics
    # =========================================================================
    logger.info("Starting finetuning with spike diagnostics...")
    start_time = time.time()

    model = DirectFineTunedTabPFNModel(cfg)
    model.fit(X, y, county_ids=county_ids)

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

    # =========================================================================
    # 4. Save results
    # =========================================================================
    model.save_to_disk(str(output_dir))

    # Save args for reproducibility
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # =========================================================================
    # 5. Print summary
    # =========================================================================
    history = model.history

    print("\n" + "=" * 80)
    print("SPIKE DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"  Total epochs trained: {len(history.train_losses)}")
    print(f"  Best epoch: {history.best_epoch}")
    print(f"  Spike count: {history.spike_count}")
    print(f"  Spike epochs: {history.spike_epochs}")
    print(f"  Output dir: {output_dir}")

    # Summarize spike records
    spike_path = output_dir / "spike_diagnostics.jsonl"
    if spike_path.exists() and spike_path.stat().st_size > 0:
        spikes = pd.read_json(spike_path, lines=True)
        print(f"\n  Spike records: {len(spikes)}")
        if len(spikes) > 0:
            print(f"  Loss range: [{spikes.step_loss.min():.2f}, {spikes.step_loss.max():.2f}]")
            print(f"  Grad norm range: [{spikes.grad_norm_before_clip.min():.4f}, "
                  f"{spikes.grad_norm_before_clip.max():.4f}]")
            print(f"  Counties involved: {spikes.fips.nunique()} unique")
            print(f"  NaN logits: {spikes.has_nan_logits.sum()} spikes")
            print(f"  Inf logits: {spikes.has_inf_logits.sum()} spikes")
            print(f"\n  Top 10 spike counties:")
            top = spikes.groupby('fips').agg(
                count=('step_loss', 'count'),
                max_loss=('step_loss', 'max'),
                mean_loss=('step_loss', 'mean'),
                mean_county_size=('county_size', 'mean'),
            ).sort_values('count', ascending=False).head(10)
            print(top.to_string(float_format='%.2f'))
    else:
        print("\n  No spikes recorded!")

    # Summarize zero-shot baseline
    zs_path = output_dir / "zeroshot_per_county.json"
    if zs_path.exists():
        with open(zs_path) as f:
            zs_losses = json.load(f)
        zs_vals = list(zs_losses.values())
        print(f"\n  Zero-shot per-county losses:")
        print(f"    n_counties: {len(zs_vals)}")
        print(f"    mean: {np.mean(zs_vals):.4f}")
        print(f"    median: {np.median(zs_vals):.4f}")
        print(f"    max: {np.max(zs_vals):.4f}")
        n_above = sum(1 for v in zs_vals if v > args.spike_threshold)
        print(f"    above spike threshold ({args.spike_threshold}): {n_above}")

        # Cross-reference: which spike counties also spike at zero-shot?
        if spike_path.exists() and spike_path.stat().st_size > 0 and len(spikes) > 0:
            spike_fips = set(spikes.fips.astype(str).unique())
            structural = {
                f: l for f, l in zs_losses.items()
                if f in spike_fips and l > args.spike_threshold
            }
            training_only = spike_fips - set(structural.keys())
            print(f"\n  Counties that spike BOTH at zero-shot and during training: "
                  f"{len(structural)}")
            if structural:
                for f, l in sorted(structural.items(), key=lambda x: -x[1])[:5]:
                    print(f"    FIPS {f}: zs_loss={l:.2f}")
            print(f"  Counties that spike ONLY during training: {len(training_only)}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("  - If spike counties also spike at zero-shot:")
    print("      → Structural issue (data/architecture). These counties are inherently hard.")
    print("  - If spike counties are fine at zero-shot:")
    print("      → Training instability. Finetuning is breaking the model for these counties.")
    print("  - If has_nan_logits=True:")
    print("      → Numerical overflow in forward pass (likely AMP/bfloat16 issue).")
    print("  - If grad_norm >> gradient_clip:")
    print("      → Gradient explosion from specific counties. Consider tighter clipping.")
    print("  - If spikes cluster on a few FIPS:")
    print("      → Specific county data distributions are pathological.")
    print("  - If spikes are uniformly distributed:")
    print("      → General training instability (lr too high, no warmup).")
    print("=" * 80)


if __name__ == "__main__":
    main()
