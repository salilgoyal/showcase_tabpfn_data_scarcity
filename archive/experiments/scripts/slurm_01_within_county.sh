#!/bin/bash
#SBATCH --job-name=calibration_no_preprocessing
#SBATCH --output=/scratch/users/salilg/property_tax/logs/within_county_%A_%a.out
#SBATCH --error=/scratch/users/salilg/property_tax/logs/within_county_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-49  # Adjust based on number of counties (N-1 where N is num counties)

# SLURM Array Job for Within-County Experiment
#
# This script runs the within-county experiment using SLURM array jobs.
# Each array task processes one county.
#
# Usage:
#   1. First, create the county registry:
#      cd /home/users/salilg/tabpfn_data_scarcity/experiments/scripts
#      python 00_create_county_registry.py
#
#   2. Check number of counties in small_county_metadata.csv
#      wc -l /home/users/salilg/tabpfn_data_scarcity/small_county_metadata.csv
#      (subtract 1 for header to get N)
#
#   3. Submit job array:
#      sbatch --array=0-N slurm_01_within_county.sh [EXPERIMENT_CONFIG]
#      where N = number of counties - 1
#      EXPERIMENT_CONFIG is optional (e.g., experiments/with_preprocessing.yaml)
#
# Example:
#   # Run with default config
#   sbatch --array=0-49 slurm_01_within_county.sh
#
#   # Run with experiment config
#   sbatch --array=0-49 slurm_01_within_county.sh experiments/with_preprocessing.yaml

# ============================================
# SHERLOCK CONFIGURATION - Edit paths here
# ============================================
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Experiment config (optional first argument)
EXPERIMENT_CONFIG="${1:-}"

# Load modules for Sherlock
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set paths
METADATA_FILE="${PROJECT_HOME}/small_county_metadata.csv"
RUNNER_SCRIPT="${PROJECT_HOME}/experiments/runners/within_county_runner.py"
# OUTPUT_DIR removed - now uses templated path from experiment config

echo "======================================"
echo "Within-County Experiment"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""

# Check if metadata file exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    echo "Please run 00_create_county_registry.py first"
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

echo "Processing County:"
echo "  FIPS: $FIPS"
echo "  Bin: $BIN_NAME"
echo "  K-folds: $K_FOLDS"
if [ -n "$EXPERIMENT_CONFIG" ]; then
    echo "  Experiment config: $EXPERIMENT_CONFIG"
fi
echo ""

# Run experiment
python "$RUNNER_SCRIPT" \
    --fips "$FIPS" \
    --bin_name "$BIN_NAME" \
    --k_folds "$K_FOLDS" \
    ${EXPERIMENT_CONFIG:+--experiment_config "$EXPERIMENT_CONFIG"}

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
