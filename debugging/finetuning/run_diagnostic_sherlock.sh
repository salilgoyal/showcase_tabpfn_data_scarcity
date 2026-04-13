#!/bin/bash
#SBATCH --job-name=ft_diagnostic
#SBATCH --output=logs/debugging/finetuning/%x_%j.out
#SBATCH --error=logs/debugging/finetuning/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# Generic SLURM script for finetuning diagnostic tests (Sherlock cluster)
#
# Usage:
#   sbatch debugging/finetuning/run_diagnostic_sherlock.sh \
#     debugging/finetuning/diag_training_spikes.py \
#     --lora_rank 8 --learning_rate 1e-4 --epoch_size 100 --max_epochs 50
#
#   sbatch debugging/finetuning/run_diagnostic_sherlock.sh \
#     debugging/finetuning/diag_training_spikes.py \
#     --spike_threshold 5 --lora_rank 4 --learning_rate 5e-5

# ============================================
# SHERLOCK CONFIGURATION
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# First argument is the Python script, rest are passed through
PYTHON_SCRIPT="${1:-}"
shift
SCRIPT_ARGS="$@"

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script required as first argument"
    echo "Usage: sbatch debugging/finetuning/run_diagnostic_sherlock.sh <script.py> [args...]"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/debugging/finetuning"

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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace authentication
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "HuggingFace token loaded"
else
    echo "Warning: HuggingFace token not found at $HOME/.cache/huggingface/token"
fi

# ============================================
# JOB INFO
# ============================================

echo "======================================"
echo "FINETUNING DIAGNOSTIC (Sherlock)"
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
