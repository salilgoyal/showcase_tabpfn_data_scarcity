#!/bin/bash
#SBATCH --job-name=gen_test_set
#SBATCH --output=logs/splits/gen_test_%j.out
#SBATCH --error=logs/splits/gen_test_%j.err
#SBATCH --partition=deho
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM job script for generating test set
# Usage: sbatch experiments/slurm/splits/generate_test_set.sh <test_config> <output_dir>
# Example: sbatch experiments/slurm/splits/generate_test_set.sh experiments/configs/test_sets/test_v1.yaml /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/

echo "======================================"
echo "GENERATING TEST SET"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Get command line arguments
TEST_CONFIG=${1:-"experiments/configs/test_sets/test_v1.yaml"}
OUTPUT_DIR=${2:-"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"}
DATA_PATH=${3:-"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet"}

echo "Test config: $TEST_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Data path: $DATA_PATH"
echo ""

# Load modules
module load python/3.12
module load devel
module load py-pyarrow/18.1.0_py312

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

# Create logs directory if needed
mkdir -p "${PROJECT_HOME}/logs/splits"

# Change to project directory
cd "$PROJECT_HOME"

# Check config file exists
if [ ! -f "$TEST_CONFIG" ]; then
    echo "ERROR: Config file not found: $TEST_CONFIG"
    exit 1
fi

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Run test set generation
echo "Running test set generation..."
echo ""

python experiments/scripts/generate_test_set.py \
    --config "$TEST_CONFIG" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Test set generated successfully!"
    echo "Review summary: ${OUTPUT_DIR}/summary_report.txt"
    echo ""
    echo "Next step: Generate train sets"
    echo "  sbatch experiments/slurm/splits/generate_train_set.sh train_v2 ${OUTPUT_DIR}"
fi

exit $EXIT_CODE
