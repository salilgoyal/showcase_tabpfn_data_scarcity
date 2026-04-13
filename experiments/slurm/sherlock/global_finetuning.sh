#!/bin/bash
#SBATCH --job-name=global_finetune
#SBATCH --output=logs/global_finetuning/sherlock/%x_%j.out
#SBATCH --error=logs/global_finetuning/sherlock/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# SLURM Job for Global TabPFN Finetuning (Sherlock)
#
# Finetunes TabPFN v2 on a large pooled dataset (10-15K samples) and
# saves the checkpoint. Single-GPU job (no array needed).
#
# Usage:
#   # External variant (non-test_v4 counties):
#   sbatch experiments/slurm/sherlock/global_finetuning.sh \
#     experiments/configs/global_finetuning/sherlock/v2_no_onehot/external_15k.yaml
#
#   # Internal variant (test_v4 train pool):
#   sbatch experiments/slurm/sherlock/global_finetuning.sh \
#     experiments/configs/global_finetuning/sherlock/v2_no_onehot/internal_15k.yaml

# ============================================
# SHERLOCK CONFIGURATION
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/sherlock/global_finetuning.sh <config_file>"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/global_finetuning/sherlock"

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
# VALIDATION
# ============================================

CONFIG_PATH="$PROJECT_HOME/$EXPERIMENT_CONFIG"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# ============================================
# JOB INFO
# ============================================

echo "======================================"
echo "EXPERIMENT TYPE: global_finetuning"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"
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
# RUN EXPERIMENT
# ============================================

cd "$PROJECT_HOME"

CMD="python $RUNNER_SCRIPT --experiment_type global_finetuning --config $EXPERIMENT_CONFIG"

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
