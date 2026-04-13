#!/bin/bash
# Quick script to test TabPFN v2.5 on an interactive node
#
# Usage:
#   1. Get interactive node: srun -p gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
#   2. Run this script: bash debugging/run_interactive_test.sh

set -e  # Exit on error

echo "======================================"
echo "TabPFN v2.5 Interactive Debug Test"
echo "======================================"
echo ""

# Load modules
echo "Loading modules..."
module load python/3.12
module load cuda
module load devel
module load cmake/3.31.4
module load py-pyarrow/18.1.0_py312
echo "✓ Modules loaded"
echo ""

# Activate venv
echo "Activating virtual environment..."
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Set environment
echo "Setting environment..."
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
echo "✓ Environment configured"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  HF_TOKEN: $(echo $HF_TOKEN | head -c 10)..."
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run test
echo "======================================"
echo "Starting test..."
echo "======================================"
cd /home/users/salilg/tabpfn_data_scarcity
python debugging/test_tabpfn_v25_interactive.py

echo ""
echo "======================================"
echo "Test completed!"
echo "======================================"
