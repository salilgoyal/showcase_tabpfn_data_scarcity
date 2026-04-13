#!/bin/bash
#SBATCH --job-name=within_county
#SBATCH --output=logs/within_county_%A_%a.out
#SBATCH --error=logs/within_county_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-49  # Adjust based on number of counties (N-1 where N is num counties)
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM Array Job for Within-County Experiment
#
# This script runs the within-county experiment using SLURM array jobs.
# Each array task processes one county.
#
# Usage:
#   1. First, create the county registry:
#      cd $PROJECT_HOME/experiments/scripts/setup
#      python create_county_registry.py
#
#   2. Check number of counties in small_county_metadata.csv
#      wc -l $PROJECT_HOME/small_county_metadata.csv
#      (subtract 1 for header to get N)
#
#   3. Submit job array:
#      sbatch --array=0-N experiments/slurm/within_county.sh [EXPERIMENT_CONFIG]
#      where N = number of counties - 1
#      EXPERIMENT_CONFIG is optional path relative to PROJECT_HOME
#
# Examples:
#   # Run with full preprocessing
#   sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/within_county/full_preprocessing.yaml
#
#   # Run with minimal preprocessing
#   sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/within_county/minimal_preprocessing.yaml
#
#   # Run calibration experiment
#   sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/calibration/tabpfn_full_preprocessing.yaml

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (optional first argument, path relative to PROJECT_HOME)
EXPERIMENT_CONFIG="${1:-}"

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
METADATA_FILE="${PROJECT_HOME}/data/small_county_metadata.csv"
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/run_experiment.py"

# Check if metadata file exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    echo "Please run experiments/scripts/setup/create_county_registry.py first"
    exit 1
fi

# Read county info from metadata file
# Skip header, get line number = TASK_ID + 2
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
COUNTY_LINE=$(sed -n "${LINE_NUM}p" "$METADATA_FILE")

if [ -z "$COUNTY_LINE" ]; then
    echo "ERROR: No data for array task ${SLURM_ARRAY_TASK_ID}"
    echo "Check that --array parameter matches number of counties"
    exit 1
fi

# Extract fields (only first 7 columns before feature_list which has commas)
FIPS=$(echo "$COUNTY_LINE" | cut -d',' -f1)
BIN_NAME=$(echo "$COUNTY_LINE" | cut -d',' -f6)
K_FOLDS=$(echo "$COUNTY_LINE" | cut -d',' -f7)

# Determine config file (use default if not provided)
if [ -z "$EXPERIMENT_CONFIG" ]; then
    EXPERIMENT_CONFIG="experiments/configs/within_county/full_preprocessing.yaml"
fi

# Extract experiment name and type from config
EXPERIMENT_NAME=$(grep -E '^\s*name:' "$PROJECT_HOME/$EXPERIMENT_CONFIG" | head -1 | sed 's/.*name:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
# Determine experiment type based on config path
if [[ "$EXPERIMENT_CONFIG" == *"calibration"* ]]; then
    EXPERIMENT_TYPE="calibration"
else
    EXPERIMENT_TYPE="within_county"
fi

echo "======================================"
echo "EXPERIMENT TYPE: ${EXPERIMENT_TYPE}"
echo "EXPERIMENT NAME: ${EXPERIMENT_NAME}"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""
echo "County Details:"
echo "  FIPS: $FIPS"
echo "  Bin: $BIN_NAME"
echo "  K-folds: $K_FOLDS"
echo "  Config: $EXPERIMENT_CONFIG"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Run experiment using unified CLI
python "$RUNNER_SCRIPT" \
    --experiment_type within_county \
    --fips "$FIPS" \
    --bin_name "$BIN_NAME" \
    --k_folds "$K_FOLDS" \
    --config "$EXPERIMENT_CONFIG"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
