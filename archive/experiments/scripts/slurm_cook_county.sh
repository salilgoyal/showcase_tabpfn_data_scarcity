#!/bin/bash
#SBATCH --job-name=cook_county_no_preprocessing
#SBATCH --output=/scratch/users/salilg/property_tax/logs/cook_county_%j.out
#SBATCH --error=/scratch/users/salilg/property_tax/logs/cook_county_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# SLURM Job for Cook County Experiment
#
# This script runs the cook county experiment using the unified framework.
#
# Usage:
#   sbatch slurm_cook_county.sh [EXPERIMENT_CONFIG]
#
# Example:
#   # Run with no preprocessing
#   sbatch slurm_cook_county.sh experiments/config/experiments/cook_county_no_preprocessing.yaml
#
#   # Run with preprocessing
#   sbatch slurm_cook_county.sh experiments/config/experiments/cook_county_with_preprocessing.yaml

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"

# Experiment config (required first argument)
EXPERIMENT_CONFIG="${1:-}"

if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config required as first argument"
    echo "Usage: sbatch slurm_cook_county.sh <config_file>"
    echo "Example: sbatch slurm_cook_county.sh experiments/config/experiments/cook_county_no_preprocessing.yaml"
    exit 1
fi

# Load modules for Sherlock
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set paths
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/runners/cook_county_runner.py"

echo "======================================"
echo "Cook County Experiment"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Experiment config: $EXPERIMENT_CONFIG"
echo ""

# Check if runner script exists
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "ERROR: Runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

# Check if config file exists
if [ ! -f "$PROJECT_HOME/$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Config file not found: $PROJECT_HOME/$EXPERIMENT_CONFIG"
    exit 1
fi

# Run experiment
python "$RUNNER_SCRIPT" "$PROJECT_HOME/$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
