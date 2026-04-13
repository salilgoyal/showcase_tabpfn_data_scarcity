#!/bin/bash
#SBATCH --job-name=finetune_tabpfn
#SBATCH --output=logs/finetuning/finetune_%j.out
#SBATCH --error=logs/finetuning/finetune_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# =============================================================================
# SLURM Job for TabPFN Fine-tuning Experiment (OLD!!!)
# =============================================================================
#
# This script runs the TabPFN fine-tuning experiment which:
# 1. Loads ~15-18M samples from multiple counties
# 2. Fine-tunes TabPFN using gradient descent with early stopping
# 3. Trains XGBoost on the same data
# 4. Evaluates both models with stratified metrics by county size
#
# Resource Requirements:
# - GPU: 1x A100 (80GB) - Required for TabPFN fine-tuning
# - Memory: 128GB - Data loading + preprocessing needs significant RAM
# - CPUs: 8 - For XGBoost parallel training and data loading
# - Time: 24 hours - Fine-tuning + XGBoost training on large data
#
# Usage:
#   sbatch experiments/slurm/finetuning/finetune_tabpfn.sh [CONFIG_FILE]
#
# Examples:
#   # Default large-scale experiment
#   sbatch experiments/slurm/finetuning/finetune_tabpfn.sh
#
#   # Custom config
#   sbatch experiments/slurm/finetuning/finetune_tabpfn.sh experiments/configs/finetuning/custom.yaml
#
# =============================================================================
# 

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Default config if not provided
DEFAULT_CONFIG="experiments/configs/finetuning/finetuning.yaml"
EXPERIMENT_CONFIG="${1:-$DEFAULT_CONFIG}"

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${SCRATCH_DIR}/logs/finetuning"

# Load modules
module load python/3.12
module load cuda/12.1
module load devel
module load cmake/3.31.4
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================
# VALIDATION
# ============================================

RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# Check if config file exists
if [ ! -f "$PROJECT_HOME/$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $PROJECT_HOME/$EXPERIMENT_CONFIG"
    exit 1
fi

# Extract experiment name from config
EXPERIMENT_NAME=$(grep -E '^\s*name:' "$PROJECT_HOME/$EXPERIMENT_CONFIG" | head -1 | sed 's/.*name:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# ============================================
# JOB INFO
# ============================================

echo "======================================"
echo "TABPFN FINE-TUNING EXPERIMENT"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""
echo "Configuration:"
echo "  Config file: $EXPERIMENT_CONFIG"
echo "  Experiment name: ${EXPERIMENT_NAME}"
echo ""
echo "Resources:"
echo "  GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Python: $(which python)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================
# RUN EXPERIMENT
# ============================================

echo "Starting experiment..."
echo "======================================"

cd "$PROJECT_HOME"

python "$RUNNER_SCRIPT" \
    --experiment_type finetuning \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

# ============================================
# COMPLETION
# ============================================

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

# Show GPU memory usage at end
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

exit $EXIT_CODE
