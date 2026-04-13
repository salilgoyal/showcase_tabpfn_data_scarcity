#!/bin/bash
#SBATCH --job-name=geo_pool_finetune
#SBATCH --output=logs/geo_pooling/nlp/finetune/%x_%A_%a.out
#SBATCH --error=logs/geo_pooling/nlp/finetune/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[32-39]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Job for Geographic Pooling + TabPFN Finetuning (NLP cluster)
#
# Per-county finetuning: for each county, fine-tunes TabPFN on that
# county's geo-pooled training set (own + neighbor data).
#
# Geo pooling training pools are typically <=500 samples, so 48GB VRAM
# is sufficient (no need for 80GB constraint used in standalone finetuning).
#
# Supports array jobs to parallelize across counties.
#
# Usage:
#   # Sequential (single job, all counties — slow):
#   sbatch experiments/slurm/nlp/geo_pooling_finetune.sh <config>
#
#   # Array job (recommended — 4 chunks):
#   sbatch --array=0-3 experiments/slurm/nlp/geo_pooling_finetune.sh \
#     experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_finetuning_k40_ratio80_droplowest5.yaml

# ============================================
# NLP CLUSTER CONFIGURATION
# ============================================
export PROJECT_HOME="/sailhome/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/property_tax"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/nlp/geo_pooling_finetune.sh <config_file>"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/geo_pooling/nlp"

# Activate conda environment
source /nlp/scr/salilg/miniconda3/bin/activate tabpfn_env

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
