#!/bin/bash
#SBATCH --job-name=geo_pool_ft
#SBATCH --output=logs/geo_pooling/sherlock/%x_%A_%a.out
#SBATCH --error=logs/geo_pooling/sherlock/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Job for Geographic Pooling + TabPFN Finetuning (Sherlock)
#
# Per-county finetuning: for each county, fine-tunes TabPFN on that
# county's geo-pooled training set (own + neighbor data).
#
# Supports array jobs to parallelize across counties.
# Each task handles a chunk of counties sequentially.
#
# Usage:
#   # Sequential (single job, all counties — slow):
#   sbatch experiments/slurm/sherlock/geo_pooling_finetune.sh <config>
#
#   # Array job (recommended — 4 chunks):
#   sbatch --array=0-3 experiments/slurm/sherlock/geo_pooling_finetune.sh \
#     experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5.yaml
#
#   # Array job (8 chunks for faster turnaround):
#   sbatch --array=0-7 experiments/slurm/sherlock/geo_pooling_finetune.sh \
#     experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5.yaml

# USE THIS TO RUN: (MAR 10 2026):
# for s in 0 1 2 3 4; do
#   sbatch --array=0-3 experiments/slurm/sherlock/geo_pooling_finetune.sh \
#     experiments/configs/geo_pooling/sherlock/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5_randsplits/test_v4_rand_s${s}_finetuning_k40_ratio80_droplowest5.yaml
# done


# ============================================
# SHERLOCK CONFIGURATION
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/sherlock/geo_pooling_finetune.sh <config_file>"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/geo_pooling/sherlock"

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
echo "EXPERIMENT TYPE: geo_pooling (finetuning)"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID} (array task: ${SLURM_ARRAY_TASK_ID:-none})"
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

CMD="python $RUNNER_SCRIPT --experiment_type geo_pooling --config $EXPERIMENT_CONFIG"

# Array job mode: pass chunk index and total number of chunks
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    COUNTY_INDEX=$SLURM_ARRAY_TASK_ID
    N_CHUNKS=${SLURM_ARRAY_TASK_COUNT}
    echo "Mode: Array job (chunk $COUNTY_INDEX of $N_CHUNKS)"
    CMD="$CMD --county_index $COUNTY_INDEX --n_chunks $N_CHUNKS"
else
    echo "Mode: Sequential (all counties in single job)"
fi

echo ""
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
