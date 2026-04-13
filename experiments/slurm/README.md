# SLURM Scripts for Sherlock Cluster

This directory contains SLURM batch scripts for running experiments on the Sherlock cluster at Stanford.

**For detailed pseudocode and experiment flow, see [Experiment Types Documentation](../docs/EXPERIMENT_TYPES.md)**

## Prerequisites

1. **Environment Setup**:
   ```bash
   # Load required modules (already in scripts)
   module load python/3.12
   module load cuda
   module load devel
   module load cmake/3.31.4
   module load py-pyarrow/18.1.0_py312

   # Activate virtual environment
   source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
   ```

2. **Project Configuration**:
   - Edit paths in each script if needed:
     - `PROJECT_HOME`: Location of project repository
     - `SCRATCH_DIR`: Scratch storage location

3. **Data Setup**:
   - Ensure county CSV files are in `/scratch/users/salilg/property_tax/county_csvs/`
   - For within-county experiments, create county registry first (see below)

## Debugging Scripts Before Submitting

Before requesting expensive cluster resources, validate your scripts to catch errors early.

**Important Note**: Both cross-county and fine-tuning experiments use the same pre-generated test/train datasets. The only difference is whether TabPFN is used in-context (cross-county) or gradient-updated (fine-tuning). XGBoost training is identical in both cases.

### Option 1: Interactive Session (Recommended)

Request an interactive node to manually test your script:

```bash
# For GPU jobs (most experiments)
srun --time=01:00:00 --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --pty bash

# For CPU-only jobs
srun --time=01:00:00 --cpus-per-task=4 --mem=16G --pty bash
```

Once in the interactive session, run these validation steps:

```bash
# 1. Navigate to project directory
cd /home/users/salilg/tabpfn_data_scarcity

# 2. Load modules (same as in your script)
module load python/3.12
module load cuda/12.1
module load devel
module load cmake/3.31.4

# 3. Activate virtual environment
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# 4. Verify environment
which python
python --version
python -c 'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}")'

# 5. Set environment variables
export PYTHONPATH="/home/users/salilg/tabpfn_data_scarcity:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 6. Verify GPU access (if using GPU)
nvidia-smi

# 7. Test Python imports
python -c "import experiments.run_experiment"
python -c "from tabpfn import TabPFNClassifier"
python -c "import xgboost"

# 8. Use existing pre-generated test/train datasets
# These datasets work for BOTH cross-county and fine-tuning experiments

# View available pre-generated splits
ls -lh /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/

# Available splits:
# - test_v1/: Test set (counties split temporally into test/train pools)
# - test_v1/train_v1/: Training set v1
# - test_v1/train_v2/: Training set v2 (larger)
# - test_v1/train_v3/: Training set v3 (largest)

# To use these in your config, add the splits section:
# splits:
#   test_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"
#   train_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/"

# 9. Validate config and run smoke test
python -c "import yaml; yaml.safe_load(open('experiments/configs/cross_county/test_v1_train_v1_smoke.yaml'))"

# Run smoke test (works for both cross-county and fine-tuning)
# NOTE: Update configs to use the splits mode if not already configured:
# splits:
#   test_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"
#   train_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/"

# For cross-county:
python experiments/run_experiment.py \
    --experiment_type cross_county \
    --config experiments/configs/cross_county/test_v1_train_v1_smoke.yaml

# For fine-tuning:
python experiments/run_experiment.py \
    --experiment_type finetuning \
    --config experiments/configs/finetuning/finetuning_smoke.yaml
```

### Option 2: Static Validation (Zero Resources)

Run these checks on the login node before requesting any resources:

```bash
# Check bash syntax
bash -n experiments/slurm/cross_county.sh
bash -n experiments/slurm/finetuning/finetune_tabpfn.sh

# Validate Python syntax
python -m py_compile experiments/run_experiment.py

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('experiments/configs/cross_county/test_v1_train_v1.yaml'))"
python -c "import yaml; yaml.safe_load(open('experiments/configs/test_sets/test_v1.yaml'))"
python -c "import yaml; yaml.safe_load(open('experiments/configs/train_sets/train_v1.yaml'))"

# Verify paths exist
test -f experiments/configs/cross_county/test_v1_train_v1.yaml && echo "Config exists" || echo "Config missing"
test -d /scratch/users/salilg/property_tax/preprocessed/cleaned_datasets/v1_no_onehot && echo "Cleaned data exists" || echo "Cleaned data missing"
```

### Option 3: Smoke Test Job (Minimal Resources)

```bash
# Smoke test configs already exist at:
# - experiments/configs/cross_county/test_v1_train_v1_smoke.yaml
# - experiments/configs/finetuning/finetuning_smoke.yaml

# Pre-generated datasets are available at:
# - /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/
# - /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/

# Ensure your config uses the splits mode:
# splits:
#   test_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"
#   train_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/"

# Submit smoke test job (5-10 min)
# For cross-county:
sbatch --time=00:15:00 --mem=16G --gres=gpu:1 \
    experiments/slurm/cross_county.sh \
    experiments/configs/cross_county/test_v1_train_v1_smoke.yaml

# For fine-tuning:
sbatch --time=00:15:00 --mem=16G --gres=gpu:1 \
    experiments/slurm/finetuning/finetune_tabpfn.sh \
    experiments/configs/finetuning/finetuning_smoke.yaml
```

### Common Validation Failures

1. **Module not found**: Check that `module load` commands succeed in interactive session
2. **Import errors**: Verify `PYTHONPATH` is set and all packages are installed
3. **Config syntax errors**: Use `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
4. **Path errors**: Check that all paths in config exist and are absolute paths
5. **GPU not detected**: Run `nvidia-smi` in interactive session to verify GPU access
6. **CUDA version mismatch**: Ensure PyTorch CUDA version matches loaded CUDA module

## Available Scripts

### 1. Data Scaling Experiments (`data_scaling.sh`)

Runs data scaling experiments that vary training data size with a fixed test set.

**Usage**:
```bash
sbatch experiments/slurm/data_scaling.sh <config_path>
```

**Examples**:
```bash
# Cook County with full preprocessing
sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_with_preprocessing.yaml

# Cook County with minimal preprocessing
sbatch experiments/slurm/data_scaling.sh experiments/configs/data_scaling/cook_county_no_preprocessing.yaml
```

**Resource Allocation**:
- Time: 1 day
- GPU: 1x
- CPUs: 4
- Memory: 16GB
- Partition: deho

### 2. Within-County Experiments (`within_county.sh`)

Runs within-county repeated k-fold cross-validation using SLURM array jobs.
Each array task processes one county.

**Setup** (first time only):
```bash
cd experiments/scripts/setup
python create_county_registry.py
```

This creates `small_county_metadata.csv` in the project root with county information.

**Usage**:
```bash
# Check number of counties
wc -l small_county_metadata.csv
# Subtract 1 for header to get N

# Submit array job for N counties
sbatch --array=0-N experiments/slurm/within_county.sh <config_path>
```

**Examples**:
```bash
# Full preprocessing (50 counties)
sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/within_county/full_preprocessing.yaml

# Minimal preprocessing
sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/within_county/minimal_preprocessing.yaml

# Calibration experiment
sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/calibration/tabpfn_full_preprocessing.yaml

# Calibration with minimal preprocessing
sbatch --array=0-49 experiments/slurm/within_county.sh experiments/configs/calibration/tabpfn_minimal.yaml
```

**Resource Allocation** (per array task):
- Time: 4 hours
- GPU: 1x
- CPUs: 4
- Memory: 16GB
- Partition: deho

### 3. Cross-County Experiments (`cross_county.sh`)

Runs cross-county generalization experiments that test how well models trained on pooled data from multiple counties generalize to a held-out target county.

**How it works**: For each county in the list, the experiment trains on data from all other counties (plus 80% of the target county) and tests on the remaining 20% of the target county. This is repeated multiple times with different random test splits.

**Usage**:
```bash
sbatch experiments/slurm/cross_county.sh <config_path>
```

**Examples**:
```bash
# Small counties cross-county experiment
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/small_in_context_10k.yaml

# Test/train set experiments
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v1.yaml
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v3.yaml
```

**Resource Allocation**:
- Time: 12 hours
- GPU: 1x
- CPUs: 4
- Memory: 32GB (larger due to pooled data)
- Partition: deho

**Notes**:
- This is **NOT** an array job - the script processes all counties sequentially
- Each county serves as target with multiple random iterations
- Handles feature mismatches automatically using feature intersection
- Total experiments = N_counties × iterations × N_models
  - Example: 9 counties × 10 iterations × 2 models = 180 experiments

### 4. Fine-tuning Experiments (`finetuning/`)

Fine-tunes TabPFN using gradient descent on pre-generated datasets and compares with XGBoost.

**Key Difference from Cross-County**: Uses the same pre-generated test/train datasets as cross-county experiments, but TabPFN is gradient-updated rather than used in-context. XGBoost training is identical. Typically uses larger datasets (e.g., test_v1_train_v1 with ~360K samples).

#### 4a. TabPFN Fine-tuning (`finetuning/finetune_tabpfn.sh`)

**Description**: Fine-tunes TabPFN using gradient descent, trains XGBoost baseline, and evaluates both models with county-stratified metrics.

**Usage**:
```bash
sbatch experiments/slurm/finetuning/finetune_tabpfn.sh [CONFIG_FILE]
```

**Examples**:
```bash
# Use test_v1_train_v1 dataset (360K samples, good for fine-tuning)
sbatch experiments/slurm/finetuning/finetune_tabpfn.sh experiments/configs/finetuning/finetuning.yaml

# Smoke test (10 tiny counties, 2 epochs)
sbatch experiments/slurm/finetuning/finetune_tabpfn.sh experiments/configs/finetuning/finetuning_smoke.yaml
```

**Resource Allocation**:
- Time: 24 hours
- GPU: 1x A100 (80GB recommended for large datasets)
- CPUs: 8
- Memory: 128GB
- Partition: gpu
- Constraint: GPU_MEM:80GB

**What it does**:
1. Loads pre-generated test/train datasets
2. Fine-tunes TabPFN with gradient descent and early stopping
3. Trains XGBoost with Optuna hyperparameter tuning
4. Evaluates both models with stratified metrics
5. Saves model checkpoints and training history

**Before running**: See "Debugging Scripts Before Submitting" section above to validate before requesting expensive resources.

#### 4b. XGBoost Only (`finetuning/train_xgboost_only.sh`)

**Description**: Trains only XGBoost for comparison or debugging purposes.

**Usage**:
```bash
sbatch experiments/slurm/finetuning/train_xgboost_only.sh [CONFIG_FILE]
```

**Resource Allocation**:
- Time: 12 hours
- GPU: 1x (optional but speeds up training)
- CPUs: 32
- Memory: 256GB
- Partition: gpu

**Use cases**:
- Running XGBoost separately from TabPFN fine-tuning
- Debugging XGBoost-specific issues
- Different resource allocation for XGBoost

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Check array job status
squeue -j <array_job_id>

# View output logs
tail -f logs/finetuning/finetune_<job_id>.out
tail -f /scratch/users/salilg/property_tax/logs/data_scaling_<job_id>.out
tail -f /scratch/users/salilg/property_tax/logs/within_county_<array_job_id>_<array_task_id>.out

# Monitor GPU usage (if you have ssh access to compute node)
ssh <node_name>  # Get from squeue output
watch -n 1 nvidia-smi

# Cancel job
scancel <job_id>

# Cancel entire array job
scancel <array_job_id>

# Cancel specific array tasks
scancel <array_job_id>_<task_id>

# Check job details
scontrol show job <job_id>

# Check job efficiency (after completion)
seff <job_id>
```

## Output Locations

Results are saved according to the `output.results_dir` specified in each config file:

- **Data Scaling**: `/scratch/users/salilg/property_tax/results/data_scaling/<experiment_name>/`
- **Within County**: `/scratch/users/salilg/property_tax/results/within_county/<experiment_name>/`
- **Cross County**: `/scratch/users/salilg/property_tax/results/cross_county/<experiment_name>/`
- **Fine-tuning**: `/scratch/users/salilg/property_tax/results/finetuning/<experiment_name>/`
- **Calibration**: `/scratch/users/salilg/property_tax/results/calibration/<experiment_name>/`

Each experiment produces:
- `results.csv` or `county_<fips>_results.csv`: Performance metrics
- `calibration.pkl` or `county_<fips>_calibration.pkl`: Calibration data (if enabled)
- `predictions.parquet`, `predictions.npy` or `county_<fips>_predictions.*`: Predictions (if enabled)
- `experiment.log`: Detailed execution log

Fine-tuning experiments also produce:
- Checkpoints in `/scratch/users/salilg/property_tax/checkpoints/finetuning/<experiment_name>/`
- Model weights and training history

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure `PYTHONPATH` is set correctly in the script
   - Check that virtual environment is activated
   - Test imports in interactive session: `python -c "import experiments.run_experiment"`

2. **Config File Not Found**:
   - Config paths are relative to `PROJECT_HOME`
   - Verify file exists: `ls $PROJECT_HOME/<config_path>`
   - Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

3. **County Metadata Not Found** (within-county only):
   - Run `experiments/scripts/setup/create_county_registry.py`
   - Check file exists: `ls $PROJECT_HOME/small_county_metadata.csv`

4. **Out of Memory**:
   - Increase `--mem` parameter in SLURM header
   - For finetuning: Reduce `batch_size` in config
   - Enable gradient accumulation: Set `gradient_accumulation_steps: 4`
   - For large counties, use `--partition=gpu` with more memory

5. **GPU Not Available**:
   - Check GPU availability: `sinfo -p deho` or `sinfo -p gpu`
   - Try different partition: `--partition=gpu`
   - For finetuning, ensure GPU constraint is available: `sinfo -p gpu -o "%N %G %m"`

6. **CUDA Out of Memory** (GPU OOM):
   - Reduce batch size in config
   - Reduce `val_batch_size` for validation
   - Enable mixed precision: `use_amp: true`
   - Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
   - Check GPU memory: `nvidia-smi`

7. **Job Fails Immediately**:
   - Check module availability: `module avail`
   - Verify virtual environment exists
   - Run validation checks from "Debugging Scripts" section
   - Check SLURM logs for specific error messages

8. **Experiment Hangs During Data Loading**:
   - Check if data files are accessible: `ls /scratch/users/salilg/property_tax/county_csvs/`
   - Verify no file system issues: `df -h /scratch/users/salilg/`
   - Check if county metadata file exists and is readable

9. **TabPFN Import Error**:
   - Ensure TabPFN is installed: `pip list | grep tabpfn`
   - Check if correct version: `python -c "import tabpfn; print(tabpfn.__version__)"`
   - Reinstall if needed: `pip install --upgrade tabpfn`

### Modifying Resource Allocation

Edit the `#SBATCH` directives at the top of each script:

```bash
#SBATCH --time=2-00:00:00     # Increase time limit
#SBATCH --mem=32G              # Increase memory
#SBATCH --cpus-per-task=8      # Increase CPUs
#SBATCH --partition=gpu        # Change partition
```

## Advanced Usage

### Running Subset of Counties

Edit `small_county_metadata.csv` to include only desired counties, then:
```bash
sbatch --array=0-N experiments/slurm/within_county.sh <config>
```

### Custom Array Job Ranges

```bash
# Run only counties 10-20
sbatch --array=10-20 experiments/slurm/within_county.sh <config>

# Run every 5th county
sbatch --array=0-49:5 experiments/slurm/within_county.sh <config>
```

### Rerunning Failed Jobs

Check logs for failed array tasks:
```bash
grep "Exit code" /scratch/users/salilg/property_tax/logs/within_county_*
```

Rerun specific tasks:
```bash
sbatch --array=5,12,28 experiments/slurm/within_county.sh <config>
```

## Quick Reference: Interactive Debugging Workflow

### For Cross-County and Fine-tuning Experiments

**Note**: Both experiment types use the same pre-generated datasets. The workflow is identical except for the `--experiment_type` flag.

**Step 1: Request interactive node**
```bash
srun --time=01:00:00 --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --pty bash
```

**Step 2: Setup environment**
```bash
cd /home/users/salilg/tabpfn_data_scarcity
module load python/3.12 cuda/12.1 devel cmake/3.31.4 py-pyarrow/18.1.0_py312
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate
export PYTHONPATH="/home/users/salilg/tabpfn_data_scarcity:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
```

**Step 3: Validate environment**
```bash
# Check Python and packages
python --version
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
python -c "import experiments.run_experiment"
nvidia-smi
```

**Step 4: Use existing pre-generated test/train datasets**
```bash
# Pre-generated datasets are available at:
# - Test set: /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/
# - Train sets: test_v1/train_v1/, test_v1/train_v2/, test_v1/train_v3/

# View available datasets
ls -lh /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/

# To use these in your config, ensure the splits section is configured:
# splits:
#   test_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"
#   train_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/"

# Optional: Generate new smoke test datasets if needed
# (Only needed if you want to create custom smoke test datasets)
# python experiments/scripts/generate_test_set.py \
#     --config experiments/configs/test_sets/test_smoke.yaml \
#     --data_path /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/data.parquet \
#     --output_dir /scratch/users/salilg/property_tax/preprocessed/test_sets/
```

**Step 5: Validate config**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('experiments/configs/cross_county/test_v1_train_v2_smoke.yaml'))"
python -c "import yaml; yaml.safe_load(open('experiments/configs/finetuning/finetuning_smoke.yaml'))"

# Verify pre-generated datasets exist
test -d /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/ && echo "Test set exists" || echo "Test set missing"
test -d /scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/ && echo "Train set exists" || echo "Train set missing"
```

**Step 6: Run smoke test experiment**
```bash
# NOTE: Ensure your config uses the splits mode to use pre-generated datasets:
# splits:
#   test_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/"
#   train_set_dir: "/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/train_v1/"

# For cross-county (in-context TabPFN, no gradient updates)
python experiments/run_experiment.py \
    --experiment_type cross_county \
    --config experiments/configs/cross_county/test_v1_train_v2_smoke.yaml

# For fine-tuning (gradient-updated TabPFN)
python experiments/run_experiment.py \
    --experiment_type finetuning \
    --config experiments/configs/finetuning/finetuning_smoke.yaml
```

**Step 7: If successful, exit and submit full job**
```bash
exit  # Exit interactive session

# Submit cross-county job
sbatch experiments/slurm/cross_county.sh experiments/configs/cross_county/test_v1_train_v2.yaml

# OR submit fine-tuning job
sbatch experiments/slurm/finetuning/finetune_tabpfn.sh experiments/configs/finetuning/finetuning.yaml
```

**Important Notes**:
- Pre-generated datasets at `/scratch/users/salilg/property_tax/preprocessed/v1_no_onehot/test_v1/` are ready to use
- Both cross-county and fine-tuning use the same test/train datasets
- The difference is only in how TabPFN is used (in-context vs. gradient-updated)
- XGBoost training is identical in both experiment types
- To use pre-generated splits, configure your YAML with the `splits:` section instead of `test_set_config:` and `train_set_config:`

## Additional Documentation

- **[Experiment Types & Pseudocode](../docs/EXPERIMENT_TYPES.md)**: Detailed execution flow for each experiment type
- **[Preprocessing Guide](../docs/README.md)**: How to configure data preprocessing
- **[Configuration Examples](../configs/)**: Example config files for different experiments
