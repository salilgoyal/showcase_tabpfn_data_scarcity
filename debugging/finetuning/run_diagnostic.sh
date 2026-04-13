#!/bin/bash
#SBATCH --job-name=ft_diagnostic
#SBATCH --output=logs/debugging/finetuning/%x_%j.out
#SBATCH --error=logs/debugging/finetuning/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# Generic SLURM script for finetuning diagnostic tests (NLP cluster)
#
# Usage:
#   # Tier 1: context sensitivity
#   sbatch debugging/finetuning/run_diagnostic.sh \
#     debugging/finetuning/diag_context_sensitivity.py \
#     --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/internal_15k/
#
#   # Tier 2: distribution swap
#   sbatch debugging/finetuning/run_diagnostic.sh \
#     debugging/finetuning/diag_distribution_swap.py \
#     --checkpoint_dir /nlp/scr/salilg/property_tax/results/global_finetuning/v2_no_onehot/internal_15k/

# ============================================
# NLP CLUSTER CONFIGURATION
# ============================================
export PROJECT_HOME="/sailhome/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/property_tax"

# First argument is the Python script, rest are passed through
PYTHON_SCRIPT="${1:-}"
shift
SCRIPT_ARGS="$@"

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script required as first argument"
    echo "Usage: sbatch debugging/finetuning/run_diagnostic.sh <script.py> [args...]"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/debugging/finetuning"

# Activate conda environment
# Initialize conda properly so that 'conda activate' can deactivate any existing env.
# (plain 'source .../activate' can silently fail if another env is already active)
eval "$(/nlp/scr/salilg/miniconda3/bin/conda shell.bash hook)"
conda activate tabpfn_env
# Hard-fail if we ended up in the wrong environment
if [[ "$(which python)" != *"tabpfn_env"* ]]; then
    echo "ERROR: conda activate tabpfn_env failed. Python is $(which python)"
    exit 1
fi

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace authentication
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "HuggingFace token loaded"
else
    echo "Warning: HuggingFace token not found"
fi

# ============================================
# JOB INFO
# ============================================

echo "======================================"
echo "FINETUNING DIAGNOSTIC"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Script: $PYTHON_SCRIPT"
echo "Args: $SCRIPT_ARGS"
echo ""
echo "Resources:"
echo "  GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Python: $(which python)"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================
# RUN DIAGNOSTIC
# ============================================

cd "$PROJECT_HOME"

CMD="python $PYTHON_SCRIPT $SCRIPT_ARGS"

echo "Command: $CMD"
echo ""

$CMD

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv

exit $EXIT_CODE
