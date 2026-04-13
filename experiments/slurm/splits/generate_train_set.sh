#!/bin/bash
#SBATCH --job-name=gen_train_set
#SBATCH --output=logs/splits/gen_train_%j.out
#SBATCH --error=logs/splits/gen_train_%j.err
#SBATCH --partition=deho
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=salilg@stanford.edu

# SLURM job script for generating train set
# Usage: sbatch experiments/slurm/splits/generate_train_set.sh <train_version> <test_split_dir>
# Example: sbatch experiments/slurm/splits/generate_train_set.sh train_v2 /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/
echo "======================================"
echo "GENERATING TRAIN SET"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Configuration
export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

# Get command line arguments
TRAIN_VERSION=${1:-"train_v2"}
TEST_SPLIT_DIR=${2:-"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"}
DATA_PATH=${3:-"/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet"}

# Construct paths
TRAIN_CONFIG="experiments/configs/train_sets/${TRAIN_VERSION}.yaml"
OUTPUT_DIR="${TEST_SPLIT_DIR}${TRAIN_VERSION}/"

echo "Train version: $TRAIN_VERSION"
echo "Train config: $TRAIN_CONFIG"
echo "Test split dir: $TEST_SPLIT_DIR"
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
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Config file not found: $TRAIN_CONFIG"
    exit 1
fi

# Check test split directory exists
if [ ! -d "$TEST_SPLIT_DIR" ]; then
    echo "ERROR: Test split directory not found: $TEST_SPLIT_DIR"
    echo "Generate test set first:"
    echo "  sbatch experiments/slurm/splits/generate_test_set.sh"
    exit 1
fi

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Run train set generation
echo "Running train set generation..."
echo ""

python experiments/scripts/generate_train_set.py \
    --config "$TRAIN_CONFIG" \
    --test_split_dir "$TEST_SPLIT_DIR" \
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
    echo "Train set generated successfully!"
    echo "Review summary: ${OUTPUT_DIR}/summary_report.txt"
    echo ""
    echo "Next step: Run experiment"
    echo "  sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_${TRAIN_VERSION}.yaml"
fi

exit $EXIT_CODE
