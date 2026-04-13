#!/bin/bash
#SBATCH --job-name=per_county_scaling_testv4
#SBATCH --output=logs/per_county_scaling/nlp/per_county_%j.out
#SBATCH --error=logs/per_county_scaling/nlp/per_county_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Job for Per-County Scaling Experiment (Sequential Mode)
#
# Processes ALL target counties sequentially in a single job.
# Counties are automatically filtered based on target_buckets in the config.
#
# Usage:
#   sbatch experiments/slurm/nlp/per_county_scaling.sh experiments/configs/per_county_scaling/nlp/test_v5.yaml

# ============================================
# NLP CLUSTER CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/sailhome/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/nlp/scr/salilg/property_tax"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-experiments/configs/per_county_scaling/nlp/test_v4.yaml}"

# Activate conda environment (NLP cluster uses conda instead of modules)
source /nlp/scr/salilg/miniconda3/bin/activate tabpfn_env

# Add project to PYTHONPATH so imports work without pip install -e .
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# HuggingFace authentication for TabPFN v2
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "HuggingFace token loaded"
else
    echo "Warning: HuggingFace token not found at $HOME/.cache/huggingface/token"
fi

# Create log directory
mkdir -p "${PROJECT_HOME}/logs/per_county_scaling/nlp"

# Set paths
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# Check if config file exists
CONFIG_PATH="$PROJECT_HOME/$EXPERIMENT_CONFIG"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "======================================"
echo "EXPERIMENT TYPE: per_county_scaling"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"
echo "Mode: Sequential (all counties)"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Run experiment for all counties (no --county_fips argument)
python "$RUNNER_SCRIPT" \
    --experiment_type per_county_scaling \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
