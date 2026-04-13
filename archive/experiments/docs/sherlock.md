Option 1: Run Single Experiment

# With preprocessing
sbatch --array=0-49 experiments/scripts/slurm_01_within_county.sh experiments/with_preprocessing.yaml

# Without preprocessing
sbatch --array=0-49 experiments/scripts/slurm_01_within_county.sh experiments/no_preprocessing.yaml

# Default (uses base_config.yaml settings)
sbatch --array=0-49 experiments/scripts/slurm_01_within_county.sh

Option 2: Run Multiple Experiments at Once

# Launch both configs for within-county experiments
cd experiments/scripts
./launch_multi_experiment.sh within_county

# Or for cross-county
./launch_multi_experiment.sh cross_county

# CALIBRATION
1. Run Calibration Experiments

  # Within-county calibration with preprocessing
  sbatch --array=0-49 experiments/scripts/slurm_01_within_county.sh \
      experiments/calibration_with_preprocessing.yaml

  # Within-county calibration without preprocessing
  sbatch --array=0-49 experiments/scripts/slurm_01_within_county.sh \
      experiments/calibration_no_preprocessing.yaml

  2. Analyze Calibration

  After jobs complete:
  # Evaluate calibration for with_preprocessing
  python experiments/analysis/evaluate_calibration.py \
      --results_dir /scratch/users/salilg/property_tax/calibration/results/calibration_with_preprocessing \
      --output_dir analysis_output/calibration_with_preprocessing

  # Evaluate calibration for no_preprocessing
  python experiments/analysis/evaluate_calibration.py \
      --results_dir /scratch/users/salilg/property_tax/calibration/results/calibration_no_preprocessing \
      --output_dir analysis_output/calibration_no_preprocessing

  3. Compare Results

  Manually compare the two calibration_aggregate.csv files:
  # View aggregate calibration
  cat analysis_output/calibration_with_preprocessing/calibration_aggregate.csv
  cat analysis_output/calibration_no_preprocessing/calibration_aggregate.csv

Output Structure

  /scratch/users/salilg/property_tax/calibration/
  ├── logs/
  │   ├── within_county_calibration_<JOBID>_<TASKID>.out
  │   └── within_county_calibration_<JOBID>_<TASKID>.err
  └── results/
      ├── calibration_with_preprocessing/
      │   ├── county_17031_results.csv              # Standard metrics
      │   ├── county_17031_calibration.pkl          # Quantile predictions + y_true
      │   └── ...
      └── calibration_no_preprocessing/
          ├── county_17031_results.csv
          ├── county_17031_calibration.pkl
          └── ...