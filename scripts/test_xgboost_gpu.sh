#!/bin/bash
#SBATCH --job-name=test_xgb_gpu
#SBATCH --output=logs/test_xgb_gpu_%j.out
#SBATCH --error=logs/test_xgb_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Test XGBoost GPU support on Sherlock

export PROJECT_HOME="/home/users/salilg/tabpfn_data_scarcity"

# Load modules
module load python/3.12
module load cuda/12.1

# Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Run test
echo "Testing XGBoost GPU support..."
echo ""
python "$PROJECT_HOME/scripts/test_xgboost_gpu.py"

echo ""
echo "Check the output above to see if GPU is working."
echo "If it failed, you may need to reinstall XGBoost with GPU support."
