"""Hyperparameter sweep: finetuning + geo pooling.

Generates configs and submits SLURM jobs for all (lr, lora_rank, epoch_size, split)
combinations. Finetuning jobs are submitted first; geo pooling jobs are submitted with
--dependency=afterok:<ft_job_id> so they start automatically when finetuning finishes.

Splits:
  --temporal        : uses test_v4/  (temporal 80/20 split)
  --seeds 0 1 2 3   : uses test_v4_rand_sN/ (random 80/20 split, seed N)

Run naming convention:
  Temporal split  : lora8_lr1e-4_ep100
  Random seed N   : lora8_lr1e-4_ep100_sN

Generated configs:
  experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep/<name>.yaml
  experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/<name>_nopooling.yaml
  experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep/<name>_ratio80.yaml

Results:
  /scratch/users/salilg/property_tax/results/global_finetuning/v2_no_onehot/sweep/<name>/
  /scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/sweep/<name>_nopooling/
  /scratch/users/salilg/property_tax/results/geo_pooling/v2_no_onehot/sweep/<name>_ratio80/

Usage (from project root on Sherlock):
    # Dry run — show all commands without submitting
    python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --dry-run

    # Full sweep: temporal + 4 random seeds (120 FT jobs + 240 geo jobs)
    python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3

    # Temporal split only (24 FT jobs + 48 geo jobs)
    python experiments/scripts/sweep.py --temporal

    # Re-submit geo pooling only (finetuning checkpoints already exist)
    python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --geo-only

    # Submit finetuning only, skip geo pooling
    python experiments/scripts/sweep.py --temporal --seeds 0 1 2 3 --ft-only
"""

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # tabpfn_data_scarcity/

FT_BASE_CONFIG = PROJECT_ROOT / (
    "experiments/configs/global_finetuning/sherlock/v2_no_onehot/"
    "internal_15k_percounty.yaml"
)
FT_SWEEP_CONFIG_DIR = (
    PROJECT_ROOT / "experiments/configs/global_finetuning/sherlock/v2_no_onehot/sweep"
)
GEO_SWEEP_CONFIG_DIR = (
    PROJECT_ROOT / "experiments/configs/geo_pooling/sherlock/v2_no_onehot/sweep"
)
FT_SLURM = PROJECT_ROOT / "experiments/slurm/sherlock/global_finetuning.sh"
GEO_SLURM = PROJECT_ROOT / "experiments/slurm/sherlock/geo_pooling.sh"

SCRATCH = "/scratch/users/salilg/property_tax"
DATA_PATH = f"{SCRATCH}/preprocessed/v2_no_onehot/"
TEMPORAL_TEST_SET = f"{SCRATCH}/preprocessed/v2_no_onehot/test_v4/"
RAND_TEST_SET_TMPL = f"{SCRATCH}/preprocessed/v2_no_onehot/test_v4_rand_s{{seed}}/"
FT_RESULTS_BASE = f"{SCRATCH}/results/global_finetuning/v2_no_onehot/sweep"
GEO_RESULTS_BASE = f"{SCRATCH}/results/geo_pooling/v2_no_onehot/sweep"

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
GRID = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
    "lora_rank":     [4, 8, 16],
    "epoch_size":    [50, 100],
}
N_SAMPLES_ALL = 999_999  # use all ~99K available training rows

# Geo pooling variants: suffix -> override dict
GEO_VARIANTS = {
    "nopooling": {"neighbor_budget_ratio": 0},
    "ratio80":   {"neighbor_budget_ratio": 0.8},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lr_str(lr: float) -> str:
    """Format LR as a short string, e.g. 1e-05 -> '1e-5'."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_name(lr: float, lora_rank: int, epoch_size: int,
             seed: Optional[int] = None) -> str:
    base = f"lora{lora_rank}_lr{lr_str(lr)}_ep{epoch_size}"
    return base if seed is None else f"{base}_s{seed}"


def make_ft_config(base: dict, lr: float, lora_rank: int, epoch_size: int,
                   seed: Optional[int]) -> dict:
    cfg = copy.deepcopy(base)
    name = run_name(lr, lora_rank, epoch_size, seed)
    results_dir = f"{FT_RESULTS_BASE}/{name}/"

    cfg["experiment"]["name"] = f"ft_sweep_{name}"
    cfg["global_finetuning"]["n_samples"] = N_SAMPLES_ALL

    ft = cfg["finetuning"]
    ft["learning_rate"] = lr
    ft["lora_rank"] = lora_rank
    ft["lora_alpha"] = float(lora_rank)  # alpha = rank → effective scale = 1
    ft["finetune_mode"] = "lora"
    ft["epoch_size"] = epoch_size

    test_set = TEMPORAL_TEST_SET if seed is None else RAND_TEST_SET_TMPL.format(seed=seed)
    cfg["splits"]["test_set_dir"] = test_set

    cfg["output"]["results_dir"] = results_dir
    cfg["output"]["checkpoint_dir"] = results_dir
    return cfg


def make_geo_config(name: str, seed: Optional[int],
                    ft_checkpoint_dir: str, variant: str) -> dict:
    test_set = TEMPORAL_TEST_SET if seed is None else RAND_TEST_SET_TMPL.format(seed=seed)
    results_dir = f"{GEO_RESULTS_BASE}/{name}_{variant}/"
    budget_ratio = GEO_VARIANTS[variant]["neighbor_budget_ratio"]

    seed_note = "" if seed is None else f", random split seed {seed}"
    return {
        "experiment": {
            "type": "geo_pooling",
            "name": f"sweep_{name}_{variant}",
            "description": f"Sweep: {name}, {variant}{seed_note}",
            "random_seed": 42,
        },
        "data": {
            "cleaned_data_path": DATA_PATH,
            "target_column": "SALE_AMOUNT",
            "project_root": "/home/users/salilg/tabpfn_data_scarcity",
        },
        "splits": {
            "test_set_dir": test_set,
        },
        "geo_pooling": {
            "centroids_csv": "data/us_county_latlng.csv",
            "max_k_neighbors": 40,
            "max_distance_miles": None,
            "neighbor_budget_ratio": budget_ratio,
            "max_samples_per_neighbor": None,
            "max_total_training_size": 10000,
            "restrict_neighbors_to_same_size_bucket": False,
        },
        "preprocessing": {
            "phase2_steps": {
                "winsorize": True,
                "winsorize_percentile": 1,
                "normalize_continuous": True,
                "impute_method": "median",
            }
        },
        "ratio_filter": {
            "enabled": True,
            "drop_bottom_percentile": 5,
            "drop_top_percentile": 0,
            "by_sale_year": True,
        },
        "models": [
            {"name": "tabpfn", "enabled": False},
            {"name": "xgboost", "enabled": False},
            {
                "name": "tabpfn_global_finetuned",
                "enabled": True,
                "checkpoint_dir": ft_checkpoint_dir,
            },
        ],
        "xgboost": {"optuna_trials": 50, "optuna_cv_folds": 3, "use_gpu": True},
        "tabpfn": {"version": "v2", "device": "cuda"},
        "checkpointing": {"enabled": True, "interval": 10, "resume": True},
        "metrics": ["r2", "mae", "rmse", "mape"],
        "output": {"results_dir": results_dir},
        "logging": {"level": "INFO"},
    }


def submit(cmd: list[str], dry_run: bool) -> Optional[str]:
    """Print and optionally run an sbatch command. Returns job ID string."""
    print(" ".join(cmd))
    if dry_run:
        return "DRY_RUN_JOB_ID"
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return None
    m = re.search(r"(\d+)", result.stdout)
    job_id = m.group(1) if m else None
    print(f"  → job {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--temporal", action="store_true",
                        help="Include temporal split (test_v4)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[],
                        metavar="N",
                        help="Random split seeds to include (e.g. --seeds 0 1 2 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--ft-only", action="store_true",
                        help="Submit only finetuning jobs (skip geo pooling)")
    parser.add_argument("--geo-only", action="store_true",
                        help="Submit only geo pooling jobs (finetuning assumed done)")
    args = parser.parse_args()

    if not args.temporal and not args.seeds:
        parser.error("Specify at least one of --temporal or --seeds N ...")
    if args.ft_only and args.geo_only:
        parser.error("--ft-only and --geo-only are mutually exclusive")

    with open(FT_BASE_CONFIG) as f:
        ft_base = yaml.safe_load(f)

    FT_SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GEO_SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of splits: None = temporal, int = random seed N
    splits: list[Optional[int]] = []
    if args.temporal:
        splits.append(None)
    splits.extend(args.seeds)

    n_hyper = (len(GRID["learning_rate"]) * len(GRID["lora_rank"])
               * len(GRID["epoch_size"]))
    n_ft = n_hyper * len(splits)
    n_geo = n_ft * len(GEO_VARIANTS)
    split_labels = ["temporal" if s is None else f"s{s}" for s in splits]
    print(f"Splits: {split_labels}")
    print(f"Grid: {len(GRID['learning_rate'])} LRs × {len(GRID['lora_rank'])} LoRA ranks × "
          f"{len(GRID['epoch_size'])} epoch sizes × {len(splits)} splits = {n_ft} FT runs")
    print(f"Geo pooling runs: {n_ft} × {len(GEO_VARIANTS)} variants = {n_geo}")
    print()

    for seed in splits:
        seed_label = "temporal" if seed is None else f"s{seed}"
        print(f"=== Split: {seed_label} ===")

        for lr in GRID["learning_rate"]:
            for lora_rank in GRID["lora_rank"]:
                for epoch_size in GRID["epoch_size"]:
                    name = run_name(lr, lora_rank, epoch_size, seed)
                    ft_checkpoint_dir = f"{FT_RESULTS_BASE}/{name}/"

                    # --- Write configs ---
                    ft_cfg = make_ft_config(ft_base, lr, lora_rank, epoch_size, seed)
                    ft_cfg_path = FT_SWEEP_CONFIG_DIR / f"{name}.yaml"
                    with open(ft_cfg_path, "w") as f:
                        yaml.dump(ft_cfg, f, default_flow_style=False, sort_keys=False)

                    geo_cfg_paths: dict[str, Path] = {}
                    for variant in GEO_VARIANTS:
                        geo_cfg = make_geo_config(name, seed, ft_checkpoint_dir, variant)
                        geo_cfg_path = GEO_SWEEP_CONFIG_DIR / f"{name}_{variant}.yaml"
                        with open(geo_cfg_path, "w") as f:
                            yaml.dump(geo_cfg, f, default_flow_style=False, sort_keys=False)
                        geo_cfg_paths[variant] = geo_cfg_path

                    # --- Submit finetuning ---
                    ft_job_id = None
                    if not args.geo_only:
                        rel = ft_cfg_path.relative_to(PROJECT_ROOT)
                        ft_job_id = submit(
                            ["sbatch", f"--job-name=ft_{name}", str(FT_SLURM), str(rel)],
                            args.dry_run,
                        )

                    # --- Submit geo pooling (with dependency if FT was just submitted) ---
                    if not args.ft_only:
                        for variant, geo_cfg_path in geo_cfg_paths.items():
                            rel = geo_cfg_path.relative_to(PROJECT_ROOT)
                            cmd = [
                                "sbatch",
                                f"--job-name=geo_{name}_{variant}",
                                "--array=0-3",
                            ]
                            if ft_job_id and not args.geo_only:
                                cmd += [f"--dependency=afterok:{ft_job_id}"]
                            cmd += [str(GEO_SLURM), str(rel)]
                            submit(cmd, args.dry_run)

    if args.dry_run:
        print("\n(dry run — nothing submitted)")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
