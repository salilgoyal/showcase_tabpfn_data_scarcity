#!/bin/bash
#SBATCH --job-name=xgboost-propertyonly-original-20seeds
#SBATCH --output=../outfiles/propertyonly-original-20seeds/xgboost.out
#SBATCH --error=../outfiles/propertyonly-original-20seeds/xgboost.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Generated from config: propertyonly-original-20seeds.yaml
# Experiment: propertyonly-original-20seeds
# Model: xgboost

echo "=================================="
echo "Starting XGBOOST Experiment"
echo "Experiment: propertyonly-original-20seeds"
echo "Start time: $(date)"
echo "=================================="

# Activate environment
module load python/3.12
module load cuda
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Create output directories
mkdir -p ../outfiles/propertyonly-original-20seeds
mkdir -p ../results/propertyonly-original-20seeds
mkdir -p ../logs/propertyonly-original-20seeds

# Run experiment
python3 run_xgboost_experiment.py \
    --config ../configs/propertyonly-original-20seeds.yaml \
    --experiment_name propertyonly-original-20seeds

exit_code=$?

echo ""
echo "=================================="
echo "XGBOOST Experiment Complete"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "=================================="

if [ $exit_code -eq 0 ]; then
    echo "✓ Results saved to: ../results/propertyonly-original-20seeds/xgboost.csv"
else
    echo "✗ Experiment failed!"
fi

exit $exit_code
