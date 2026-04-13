# Reorganization Complete! 🎉

The repository has been successfully reorganized into a clean, maintainable structure.

## New Structure

```
tabpfn_data_scarcity/
├── src/                        # ✅ Core reusable library
│   ├── models/                # Model wrappers (TabPFN, XGBoost)
│   ├── data/                  # Data loading & preprocessing
│   ├── evaluation/            # Metrics and aggregation
│   └── utils/                 # Utilities
│
├── experiments/                # ✅ Experiment code
│   ├── runners/               # Base runner + helpers
│   │   └── base_runner.py    # Shared experiment logic
│   ├── experiment_types/      # Experiment strategies
│   │   └── data_scaling.py   # Replaces cook_county_runner
│   ├── configs/               # All experiment configs
│   ├── scripts/               # Utility scripts
│   ├── analysis/              # Analysis utilities
│   └── run_experiment.py      # Main CLI entry point
│
├── notebooks/                  # ✅ All analysis notebooks
│   ├── within_county/
│   ├── data_scaling/
│   ├── model_comparison/
│   └── exploratory/
│
├── results/                    # ✅ Experiment results
│
├── archive/                    # ✅ Old code (can delete after testing)
│   ├── experiments/           # Old experiment code
│   ├── evelyn_files/          # Old preprocessing location
│   ├── cook_county_analysis/  # Old Cook County code
│   └── ...
│
├── setup.py                    # ✅ Makes src/ installable
├── requirements.txt
└── README.md
```

---

## What Changed

### ✅ **Consolidated Preprocessing**
- **Old**: `evelyn_files/` (separate, awkward)
- **New**: `src/data/preprocess.py` (integrated)

### ✅ **Unified Library Code**
- **Old**: Scattered in `experiments/models/`, `experiments/data/`, `experiments/evaluation/`
- **New**: Centralized in `src/` (importable from anywhere)

### ✅ **Cleaner Experiments**
- **Old**: `cook_county_runner.py` with duplicated logic
- **New**: `experiment_types/data_scaling.py` using base runner

### ✅ **Organized Configs**
- **Old**: `experiments/config/`
- **New**: `experiments/configs/` with better organization

### ✅ **No More Root Clutter**
- **Old**: Scripts scattered at root level
- **New**: All in `experiments/scripts/`

---

## How to Use

### Set Up Imports (PYTHONPATH method)

**No need to run `pip install -e .`** - it can cause dependency conflicts in your environment.

Instead, the SLURM scripts automatically set `PYTHONPATH`:
```bash
export PYTHONPATH="${PROJECT_HOME}:${PYTHONPATH}"
```

For interactive use, set it manually:
```bash
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH
```

Now you can import from anywhere:
```python
from src.models import TabPFNModel, XGBoostModel
from src.data import CountyDataLoader, load_and_prepare_data
from src.evaluation import compute_metrics
```

### Run Experiments

```bash
# Data scaling experiment (new unified CLI)
python experiments/run_experiment.py \
  --experiment_type data_scaling \
  --config experiments/configs/experiments/data_scaling/cook_county_example.yaml

# Within-county CV (backward compatible)
python experiments/run_experiment.py \
  --experiment_type within_county \
  --fips 1011 \
  --bin_name small \
  --k_folds 5 \
  --config experiments/configs/experiments/with_preprocessing.yaml
```

### Create Notebooks

```bash
# Start Jupyter in notebooks/
cd notebooks
jupyter notebook
```

Import works cleanly:
```python
from src.models import TabPFNModel
from src.data import CountyDataLoader
```

---

## Next Steps

### Phase 1: Test (Do Now)
1. Test that data_scaling experiment runs:
   ```bash
   cd experiments
   python run_experiment.py \
     --experiment_type data_scaling \
     --config configs/experiments/data_scaling/cook_county_example.yaml \
     --output_dir ../results/test_run
   ```

2. Verify imports work:
   ```bash
   python -c "from src.models import TabPFNModel; print('✓ Imports work!')"
   ```

### Phase 2: Migrate Notebooks (Later)
1. Move notebooks from `archive/notebooks/` to `notebooks/`
2. Update their imports to use `from src.*`
3. Organize by experiment type

### Phase 3: Clean Up (When Ready)
1. After verifying everything works, delete `archive/`:
   ```bash
   rm -rf archive/
   ```

2. Update `.gitignore` to ignore `results/` and large data files

---

## Benefits Achieved

### 1. ✅ **Clean Imports**
```python
# Before: 😱
sys.path.insert(0, '../../evelyn_files')
from preprocess import Preprocess

# After: 😊
from src.data.preprocess import Preprocess
```

### 2. ✅ **No Code Duplication**
- Models: **1 location** (was 3)
- Preprocessing: **1 location** (was 2)
- Evaluation: **1 location** (was 3)

### 3. ✅ **Extensible Framework**
Adding new experiment types is now easy:
- Inherit from `BaseExperimentRunner`
- Implement data splitting logic
- ~100 lines vs ~300 lines before

### 4. ✅ **Installable Package**
```bash
pip install -e .  # Install in dev mode
# Now import from anywhere!
```

### 5. ✅ **Better Organization**
- Clear separation: library vs experiments vs analysis
- Easy to find things
- Consistent structure

### 6. ✅ **SLURM Integration**
- Centralized batch scripts in `slurm/`
- All scripts use unified `run_experiment.py` CLI
- Comprehensive documentation in `slurm/README.md`
- Clean log organization (local `logs/`, cluster `/scratch/`)

---

## SLURM Usage on Cluster

### Running Experiments

**Within-County Cross-Validation:**
```bash
# First, create county registry
cd $PROJECT_HOME/experiments/scripts
python 00_create_county_registry.py

# Check number of counties
wc -l $PROJECT_HOME/small_county_metadata.csv

# Submit array job (adjust --array based on county count)
sbatch --array=0-49 slurm/within_county.sh experiments/configs/experiments/with_preprocessing.yaml
```

**Data Scaling Experiments:**
```bash
# Run Cook County or any other data scaling experiment
sbatch slurm/data_scaling.sh experiments/configs/experiments/data_scaling/cook_county_example.yaml
```

### Monitoring Jobs

```bash
# View your jobs
squeue -u $USER

# Check logs (cluster)
tail -f /scratch/users/salilg/property_tax/logs/within_county_*.out
tail -f /scratch/users/salilg/property_tax/logs/data_scaling_*.err

# Job accounting
sacct -u $USER --starttime=today
```

See `slurm/README.md` for complete documentation, troubleshooting, and examples.

---

## File Mapping (Where Things Went)

| Old Location | New Location | Notes |
|-------------|--------------|-------|
| `evelyn_files/` | `src/data/` | Preprocessing integrated |
| `experiments/models/` | `src/models/` | Library code |
| `experiments/data/` | `src/data/` | Library code |
| `experiments/evaluation/` | `src/evaluation/` | Library code |
| `count_county_rows.py` (root) | `experiments/scripts/` | Utility scripts |
| `experiments/config/` | `experiments/configs/` | Renamed for clarity |
| `cook_county_runner.py` | `experiment_types/data_scaling.py` | Refactored |
| `archive/experiments/scripts/slurm_*.sh` | `slurm/` | Centralized, updated to use new CLI |

---

## Troubleshooting

### Import errors?
```bash
# Set PYTHONPATH (SLURM scripts do this automatically)
export PYTHONPATH=/home/users/salilg/tabpfn_data_scarcity:$PYTHONPATH

# Verify imports work
python -c "from src.models import TabPFNModel; print('✓ Imports work!')"
```

### Old runner files referenced?
Check `archive/experiments/runners/` for the old code if needed for reference.

### Need the old preprocessing?
It's in `archive/evelyn_files/` - same code, now also in `src/data/`

---

## Summary

**✅ Reorganization Complete!**

You now have:
- Clean, organized codebase
- Reusable library code in `src/`
- Unified experiment framework
- Installable package
- Centralized SLURM scripts in `slurm/`
- Clean log and results organization
- Clear structure for growth

**Next Steps:**
1. **Test Imports**: Verify `export PYTHONPATH=...` and test `from src.models import TabPFNModel`
2. **Run Test Job**: Submit a small SLURM job to verify everything works
3. **Migrate Notebooks**: Move analysis notebooks from `archive/notebooks/` when ready
4. **Clean Up**: Delete `archive/` once confident

**Documentation:**
- Full reorganization plan: `REORGANIZATION_PLAN.md`
- SLURM usage and troubleshooting: `slurm/README.md`
- Experiment framework: `experiments/REFACTORING_GUIDE.md`
