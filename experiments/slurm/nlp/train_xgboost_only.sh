#!/bin/bash
#SBATCH --job-name=xgb_gpu_test_v4_train_v7
#SBATCH --output=logs/finetuning/xgboost_%j.out
#SBATCH --error=logs/finetuning/xgboost_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=salilg@stanford.edu

# =============================================================================
# SLURM Job for XGBoost Training (Large-Scale)
# =============================================================================
#
# This script trains XGBoost only, for cases where:
# - You want to run XGBoost separately from TabPFN fine-tuning
# - You're debugging XGBoost specifically
# - You want different resource allocation for XGBoost
#
# Resource Requirements:
# - GPU: 1 (optional but speeds up training)
# - Memory: 256GB - XGBoost on 15M+ rows needs significant RAM
# - CPUs: 32 - Histogram building is CPU-parallel
# - Time: 12 hours - Optuna tuning + final training
#
# Usage:
#   sbatch experiments/slurm/finetuning/train_xgboost_only.sh [CONFIG_FILE]
#
# =============================================================================

export PROJECT_HOME="/sailhome/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/property_tax"

DEFAULT_CONFIG="experiments/configs/finetuning/large_scale.yaml"
EXPERIMENT_CONFIG="${1:-$DEFAULT_CONFIG}"

# Create log directory
mkdir -p "${SCRATCH_DIR}/logs/finetuning"

# Activate conda environment (NLP cluster uses conda instead of modules)
source /nlp/scr/salilg/miniconda3/bin/activate tabpfn_env

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Check config
if [ ! -f "$PROJECT_HOME/$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $PROJECT_HOME/$EXPERIMENT_CONFIG"
    exit 1
fi

echo "======================================"
echo "XGBOOST TRAINING (LARGE-SCALE)"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config: $EXPERIMENT_CONFIG"
echo ""

cd "$PROJECT_HOME"

# Run with XGBoost only (disable TabPFN via command override)
# Note: You may need to modify the config or add a --models flag
python "$PROJECT_HOME/experiments/run_experiment.py" \
    --experiment_type finetuning \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
