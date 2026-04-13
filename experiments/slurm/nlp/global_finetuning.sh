#!/bin/bash
#SBATCH --job-name=global_finetune
#SBATCH --output=logs/global_finetuning/nlp/%x_%j.out
#SBATCH --error=logs/global_finetuning/nlp/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[32-39]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# SLURM Job for Global TabPFN Finetuning (NLP cluster)
#
# Finetunes TabPFN v2 on a large pooled dataset (10-15K samples) and
# saves the checkpoint. Single-GPU job (no array needed).
#
# Usage:
#   # External variant (non-test_v4 counties):
#   sbatch experiments/slurm/nlp/global_finetuning.sh \
#     experiments/configs/global_finetuning/nlp/v2_no_onehot/external_15k.yaml
#
#   # Internal variant (test_v4 train pool):
#   sbatch experiments/slurm/nlp/global_finetuning.sh \
#     experiments/configs/global_finetuning/nlp/v2_no_onehot/internal_15k.yaml

# ============================================
# NLP CLUSTER CONFIGURATION
# ============================================
export PROJECT_HOME="/sailhome/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/property_tax"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/nlp/global_finetuning.sh <config_file>"
    exit 1
fi

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/global_finetuning/nlp"

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
