#!/bin/bash
#SBATCH --job-name=single_county_scaling
#SBATCH --output=/sailhome/salilg/showcase_tabpfn_data_scarcity/logs/single_county_scaling/nlp/%x/%j.out
#SBATCH --error=/sailhome/salilg/showcase_tabpfn_data_scarcity/logs/single_county_scaling/nlp/%x/%j.err
#SBATCH --time=2-00:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilslurm@gmail.com

# Single-County Data Scaling Experiment
#
# Runs learning curves for a single county (e.g., Cook County):
# sweeps training sizes x seeds x models.
#
# Usage:
#   sbatch experiments/slurm/nlp/single_county_scaling.sh <config_file>
#
# Example:
#   sbatch experiments/slurm/nlp/single_county_scaling.sh \
#     experiments/configs/single_county_scaling/nlp/v2_no_onehot/cook_county_20seeds.yaml

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
    echo "Usage: sbatch experiments/slurm/nlp/single_county_scaling.sh <config_file>"
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
mkdir -p "${PROJECT_HOME}/logs/single_county_scaling/nlp"

# Check config exists
CONFIG_PATH="$PROJECT_HOME/$EXPERIMENT_CONFIG"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

echo "======================================"
echo "EXPERIMENT TYPE: single_county_scaling"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"
echo ""

cd "$PROJECT_HOME"

CMD="python $RUNNER_SCRIPT --experiment_type single_county_scaling --config $EXPERIMENT_CONFIG $EXTRA_ARGS"

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
