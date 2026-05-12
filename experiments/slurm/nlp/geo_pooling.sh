#!/bin/bash
#SBATCH --job-name=geo_pooling
#SBATCH --output=/sailhome/salilg/showcase_tabpfn_data_scarcity/logs/geo_pooling/nlp/%x/%A_%a.out
#SBATCH --error=/sailhome/salilg/showcase_tabpfn_data_scarcity/logs/geo_pooling/nlp/%x/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# SLURM Job for Geographic Pooling Experiment
#
# Supports two modes:
#   1. Sequential: process all counties in a single job (default)
#   2. Array job: split counties into N_CHUNKS chunks, one per array task
#
# Usage:
#   # Sequential (single job, all counties):
#   sbatch experiments/slurm/nlp/geo_pooling.sh <config>
#
#   # Array job (split into 4 chunks):
#   N_CHUNKS=4 sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh <config>
#
# 4 chunks — Python figures out the size (ceil(n_counties / 4))
# sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh experiments/configs/geo_pooling/nlp/test_v4_tiny.yaml

# # 8 chunks
# sbatch --array=0-7 experiments/slurm/nlp/geo_pooling.sh experiments/configs/geo_pooling/nlp/test_v4_all.yaml

# LATEST
# sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k1K_ratio10K_droplowest5.yaml

# Rand splits
# for S in 0 1 2 3 4; do
#   sbatch --array=0-3 experiments/slurm/nlp/geo_pooling.sh \
#     experiments/configs/geo_pooling/nlp/v2_no_onehot/test_v4_k40_ratio80_droplowest5_global_finetuned_internal_percounty_randsplits/test_v4_rand_s${S}_k40_ratio80_droplowest5_global_finetuned_internal_percounty.yaml
# done


# ============================================
# NLP CLUSTER CONFIGURATION
# ============================================
export PROJECT_HOME="/sailhome/salilg/showcase_tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/showcase_property_tax"

# Experiment config (required first argument); any extra args (e.g. --models) are passed through
EXPERIMENT_CONFIG="${1:-}"
EXTRA_ARGS="${@:2}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/nlp/geo_pooling.sh <config_file>"
    exit 1
fi

# Activate conda environment
source /nlp/scr/salilg/miniconda3/bin/activate tabpfn_env

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# HuggingFace authentication
# Use NLP local home (parent of PROJECT_HOME) rather than $HOME, which in SLURM
# resolves to the AFS home and won't have the token saved by huggingface-cli login.
NLP_HOME=$(dirname "$PROJECT_HOME")
HF_TOKEN_FILE="$NLP_HOME/.cache/huggingface/token"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    echo "HuggingFace token loaded from $HF_TOKEN_FILE"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "HuggingFace token loaded from \$HOME cache"
else
    echo "Warning: HuggingFace token not found (checked $HF_TOKEN_FILE and \$HOME/.cache/huggingface/token)"
fi

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/geo_pooling/nlp"

# Check config exists
CONFIG_PATH="$PROJECT_HOME/$EXPERIMENT_CONFIG"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

echo "======================================"
echo "EXPERIMENT TYPE: geo_pooling"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID} (array task: ${SLURM_ARRAY_TASK_ID:-none})"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"

cd "$PROJECT_HOME"

# Build command
CMD="python $RUNNER_SCRIPT --experiment_type geo_pooling --config $EXPERIMENT_CONFIG $EXTRA_ARGS"

# Array job mode: pass chunk index and total number of chunks
# SLURM_ARRAY_TASK_COUNT is set automatically by SLURM (= number of tasks in --array)
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

exit $EXIT_CODE
