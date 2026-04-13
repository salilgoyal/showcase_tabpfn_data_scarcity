#!/bin/bash
#SBATCH --job-name=analyze_preprocessed
#SBATCH --output=preprocessing/logs/analyze_%j.out
#SBATCH --error=preprocessing/logs/analyze_%j.err
#SBATCH --partition=deho
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM job script for analyzing preprocessed data
# Usage: sbatch preprocessing/slurm/analyze_data.sh <data_path> <output_dir>

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Get command line arguments
DATA_PATH=${1:-"/scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot/data.parquet"}
OUTPUT_DIR=${2:-"preprocessing/analysis/v1_no_onehot/"}

# Load modules
module load python/3.12
module load devel
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH (for consistency and future-proofing)
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create logs and output directories if needed
mkdir -p "${PROJECT_HOME}/preprocessing/logs"
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "======================================"
echo "PREPROCESSING DATA ANALYSIS"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Memory requested: ${SLURM_MEM_PER_NODE}MB"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK}"
echo ""

# Change to project directory
cd "$PROJECT_HOME"

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Run analysis script
echo "Running analysis..."
echo ""

python preprocessing/scripts/analyze_preprocessed_data.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE
