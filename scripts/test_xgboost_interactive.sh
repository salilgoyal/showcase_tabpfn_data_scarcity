#!/bin/bash
# Interactive XGBoost GPU Test Script
#
# This script helps you test XGBoost training interactively to catch errors quickly
# without waiting in SLURM queue
#
# Usage:
#   1. Request interactive GPU node:
#      srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#
#   2. Run this script:
#      bash scripts/test_xgboost_interactive.sh

set -e  # Exit on error

export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"
export SCRATCH_DIR="/scratch/users/salilg/property_tax"

echo "======================================"
echo "INTERACTIVE XGBOOST TEST"
echo "======================================"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo ""

# Load modules
echo "Loading modules..."
module load python/3.12
module load cuda/12.1
module load py-pyarrow/18.1.0_py312
module load py-xgboost/3.0.0_py312

# Activate virtual environment
echo "Activating virtual environment..."
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"

cd "$PROJECT_HOME"

# Check GPU availability
echo ""
echo "Checking GPU..."
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Run smoke test
echo ""
echo "======================================"
echo "Running XGBoost smoke test..."
echo "Config: experiments/configs/finetuning/xgboost_smoke_test.yaml"
echo "======================================"
echo ""

python experiments/run_experiment.py \
    --experiment_type finetuning \
    --config experiments/configs/finetuning/xgboost_smoke_test.yaml

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Success! You can now run the full experiment:"
    echo "  sbatch experiments/slurm/finetuning/train_xgboost_only.sh"
else
    echo ""
    echo "✗ Test failed. Fix errors above before submitting to queue."
fi

exit $EXIT_CODE
