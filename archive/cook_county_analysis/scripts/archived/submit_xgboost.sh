#!/bin/bash
#SBATCH --job-name=xgboost-propertyonly-original-20seeds
#SBATCH --output=../outfiles/propertyonly-original-20seeds/xgboost.out
#SBATCH --error=../outfiles/propertyonly-original-20seeds/xgboost.err
#SBATCH --time=10:00:00
#SBATCH --partition=deho
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

###############################
# Submit XGBoost experiment
# Usage: sbatch submit_xgboost.sh <experiment_name>
# Example: sbatch submit_xgboost.sh propertyonly-original-20seeds
###############################

EXPERIMENT_NAME=${1:-"propertyonly-original-20seeds"}

echo "=================================="
echo "Starting XGBoost Experiment"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=================================="

# Activate environment
# For NLP cluster (commented out):
# . /nlp/scr/salilg/miniconda3/etc/profile.d/conda.sh
# conda activate pfn_env

# For Sherlock:
module load python/3.12.1
source ../../.venv/bin/activate

# Create output directories
mkdir -p ../outfiles/$EXPERIMENT_NAME
mkdir -p ../results/$EXPERIMENT_NAME
mkdir -p ../logs/$EXPERIMENT_NAME

# Run experiment
python3 run_xgboost_experiment.py \
    --experiment_name $EXPERIMENT_NAME \
    --data_path ../data/cook_county.csv \
    --output_dir ../results \
    --no_predictions

exit_code=$?

echo ""
echo "=================================="
echo "XGBoost Experiment Complete"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "=================================="

if [ $exit_code -eq 0 ]; then
    echo "✓ Results saved to: ../results/$EXPERIMENT_NAME/xgboost.csv"
else
    echo "✗ Experiment failed!"
fi

exit $exit_code
