#!/bin/bash
#SBATCH --job-name=cross_county_gpu_test_v5_train_v1
#SBATCH --output=logs/cross_county/cross_county_%j.out
#SBATCH --error=logs/cross_county/cross_county_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Job for Cross-County Generalization Experiment
#
# This script runs cross-county experiments that test how well models trained
# on pooled data from multiple counties generalize to held-out target counties.
#
# The experiment loops over all counties internally, training on all-but-one
# and testing on the held-out county. This is NOT an array job because the
# experiment handles county iteration internally.
#
# Usage:
#   sbatch experiments/slurm/cross_county.sh <EXPERIMENT_CONFIG>
#
# Examples:
#   # Small counties cross-county experiment
#   sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/small_counties.yaml
#
#   # Override county list at runtime (optional)
#   python experiments/run_experiment.py \
#       --experiment_type cross_county \
#       --config experiments/configs/cross_county/small_counties.yaml \
#       --county_list "1011,1041,1065"

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (required first argument, path relative to PROJECT_HOME)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/cross_county.sh <config_file>"
    echo "Example: sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/small_counties.yaml"
    exit 1
fi

# Load modules for Sherlock
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH so imports work without pip install -e .
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# HuggingFace authentication for TabPFN v2.5
# Read token from the standard HuggingFace cache location
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "✓ HuggingFace token loaded"
else
    echo "⚠ Warning: HuggingFace token not found at $HOME/.cache/huggingface/token"
fi

# Set paths
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# Check if config file exists
if [ ! -f "$PROJECT_HOME/$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $PROJECT_HOME/$EXPERIMENT_CONFIG"
    exit 1
fi

# Extract experiment name from config
EXPERIMENT_NAME=$(grep -E '^\s*name:' "$PROJECT_HOME/$EXPERIMENT_CONFIG" | head -1 | sed 's/.*name:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Count number of counties in config
NUM_COUNTIES=$(grep -E '^\s*-\s+[0-9]+\s*$' "$PROJECT_HOME/$EXPERIMENT_CONFIG" | wc -l)

echo "======================================"
echo "EXPERIMENT TYPE: cross_county"
echo "EXPERIMENT NAME: ${EXPERIMENT_NAME}"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"
echo "Number of counties: $NUM_COUNTIES"
echo ""
echo "Note: This job processes all counties sequentially"
echo "      Each county will be used as target with multiple iterations"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Run experiment using unified CLI
python "$RUNNER_SCRIPT" \
    --experiment_type cross_county \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
