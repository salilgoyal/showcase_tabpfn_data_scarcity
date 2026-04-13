#!/usr/bin/env python3
"""Plot training loss curves from sweep experiments."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_run(run_dir, save_path=None):
    """Plot training curves for a single run."""
    history_file = Path(run_dir) / "training_history.json"

    if not history_file.exists():
        print(f"No history file found at {history_file}")
        return

    with open(history_file) as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Training Curves: {Path(run_dir).name}", fontsize=16)

    # Training and validation loss
    ax = axes[0, 0]
    ax.plot(history['train_losses'], label='Train Loss', marker='o', markersize=3)
    ax.plot(history['val_losses'], label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation R2
    ax = axes[0, 1]
    ax.plot(history['val_r2'], label='Val R2', color='green', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation MAE
    ax = axes[1, 0]
    ax.plot(history['val_mae'], label='Val MAE', color='red', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Validation MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(history['learning_rates'], label='Learning Rate', color='purple', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_multiple_runs(base_dir, pattern="lora16_lr1e-4*", save_dir=None):
    """Plot training curves for multiple runs matching a pattern."""
    base_path = Path(base_dir)
    runs = sorted(base_path.glob(pattern))

    if not runs:
        print(f"No runs found matching pattern '{pattern}' in {base_dir}")
        return

    print(f"Found {len(runs)} runs matching '{pattern}'")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparison: {pattern}", fontsize=16)

    for run_dir in runs:
        history_file = run_dir / "training_history.json"
        if not history_file.exists():
            continue

        with open(history_file) as f:
            history = json.load(f)

        label = run_dir.name

        # Training loss
        axes[0, 0].plot(history['train_losses'], label=label, alpha=0.7)

        # Validation loss
        axes[0, 1].plot(history['val_losses'], label=label, alpha=0.7)

        # Validation R2
        axes[1, 0].plot(history['val_r2'], label=label, alpha=0.7)

        # Validation MAE
        axes[1, 1].plot(history['val_mae'], label=label, alpha=0.7)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Validation R²')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f"comparison_{pattern.replace('*', 'all')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Path to a single run directory")
    parser.add_argument("--base-dir", default="/scratch/users/salilg/property_tax/results/global_finetuning/v2_no_onehot/sweep",
                        help="Base directory containing sweep runs")
    parser.add_argument("--pattern", default="lora16_lr1e-4*",
                        help="Pattern to match run directories (e.g., 'lora16_lr1e-4*')")
    parser.add_argument("--save-dir", help="Directory to save plots (optional)")
    parser.add_argument("--mode", choices=["single", "compare"], default="compare",
                        help="Plot a single run or compare multiple runs")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.run:
            parser.error("--run is required for single mode")
        plot_run(args.run, args.save_dir)
    else:
        plot_multiple_runs(args.base_dir, args.pattern, args.save_dir)
