#!/bin/bash
#SBATCH --job-name=data_scaling
#SBATCH --output=logs/data_scaling_%j.out
#SBATCH --error=logs/data_scaling_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Job for Data Scaling Experiment (formerly Cook County)
#
# This script runs data scaling experiments (varying training data size).
# Can be used for Cook County or any other county/dataset.
#
# Usage:
#   sbatch experiments/slurm/data_scaling.sh <EXPERIMENT_CONFIG>
#
# Examples:
#   # Cook County with preprocessing
#   sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_with_preprocessing.yaml
#
#   # Cook County without preprocessing
#   sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_no_preprocessing.yaml

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (required first argument, path relative to PROJECT_HOME)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch experiments/slurm/data_scaling.sh <config_file>"
    echo "Example: sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_with_preprocessing.yaml"
    exit 1
fi

# Load modules for Sherlock
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH so imports work without pip install -e .
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Set paths
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# Check if config file exists
if [ ! -f "$PROJECT_HOME/$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $PROJECT_HOME/$EXPERIMENT_CONFIG"
    exit 1
fi

# Extract experiment name from config
EXPERIMENT_NAME=$(grep -E '^\s*name:' "$PROJECT_HOME/$EXPERIMENT_CONFIG" | head -1 | sed 's/.*name:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

echo "======================================"
echo "EXPERIMENT TYPE: data_scaling"
echo "EXPERIMENT NAME: ${EXPERIMENT_NAME}"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Config file: $EXPERIMENT_CONFIG"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Run experiment using unified CLI
python "$RUNNER_SCRIPT" \
    --experiment_type data_scaling \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
