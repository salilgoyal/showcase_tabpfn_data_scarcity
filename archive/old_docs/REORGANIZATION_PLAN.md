# Repository Reorganization Plan

## Philosophy

**Separate concerns clearly:**
- `src/` - Reusable library code (models, data utils, evaluation)
- `experiments/` - Experiment-specific code (runners, configs, scripts)
- `notebooks/` - All analysis notebooks
- `results/` - All experiment results (or symlink to /scratch/)
- `archive/` - Deprecated code

## Proposed Structure

```
tabpfn_data_scarcity/
├── README.md
├── requirements.txt
├── setup.py                    # NEW: Make src/ installable
│
├── src/                        # NEW: Core library (reusable code)
│   ├── __init__.py
│   ├── models/                 # From experiments/models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── tabpfn_wrapper.py
│   │   └── xgboost_wrapper.py
│   ├── data/                   # From experiments/data/ + evelyn_files/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessing.py   # Consolidated preprocessing
│   │   ├── column_definitions.py
│   │   └── splitters.py
│   ├── evaluation/             # From experiments/evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── aggregation.py
│   └── utils/                  # NEW: Utilities
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
│
├── experiments/                # Experiment-specific code
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── base_runner.py
│   │   └── experiment_types/
│   │       ├── __init__.py
│   │       ├── within_county.py
│   │       ├── cross_county.py
│   │       ├── data_scaling.py
│   │       └── in_context_pooling.py
│   ├── configs/
│   │   ├── base_config.yaml
│   │   ├── within_county_config.yaml
│   │   ├── cross_county_config.yaml
│   │   └── experiments/        # Specific experiment configs
│   │       ├── within_county/
│   │       ├── data_scaling/
│   │       └── pooling/
│   ├── scripts/                # Utility scripts
│   │   ├── create_county_registry.py
│   │   ├── aggregate_results.py
│   │   ├── count_county_rows.py      # From root
│   │   ├── save_counties.py          # From root
│   │   └── slurm/                    # SLURM job scripts
│   │       ├── within_county.sh
│   │       ├── cross_county.sh
│   │       └── data_scaling.sh
│   ├── analysis/               # Analysis utilities
│   │   ├── __init__.py
│   │   ├── model_comparison_plots.py
│   │   └── evaluate_calibration.py
│   ├── run_experiment.py       # Main CLI
│   └── docs/                   # Experiment documentation
│       ├── QUICKSTART.md
│       ├── REFACTORING_GUIDE.md
│       └── PREPROCESSING_GUIDE.md
│
├── notebooks/                  # ALL analysis notebooks
│   ├── within_county/
│   │   └── analysis.ipynb
│   ├── data_scaling/
│   │   └── learning_curves.ipynb
│   ├── model_comparison/
│   │   └── tabpfn_vs_xgb.ipynb
│   └── exploratory/
│       └── data_exploration.ipynb
│
├── results/                    # All experiment results
│   ├── within_county/
│   ├── cross_county/
│   ├── data_scaling/
│   └── pooling/
│   # OR: symlink to /scratch/users/salilg/property_tax/results/
│
├── data/                       # Data files (optional)
│   ├── README.md              # Explains data is on /scratch/
│   └── .gitkeep
│   # OR: symlink to /scratch/users/salilg/property_tax/
│
├── tests/                      # NEW: Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_evaluation.py
│
└── archive/                    # Deprecated code
    ├── cook_county_analysis/
    ├── prior_posterior_experiments/
    ├── evelyn_files/           # Keep for reference
    └── old_results/
```

---

## Migration Steps

### Phase 1: Create Core Library (src/)

**Why**: Separates reusable code from experiment-specific code. Makes code importable and testable.

```bash
# 1. Create src/ structure
mkdir -p src/{models,data,evaluation,utils}

# 2. Move experiments/models/ → src/models/
mv experiments/models/*.py src/models/

# 3. Consolidate preprocessing
# Move evelyn_files/preprocess.py → src/data/preprocessing_core.py
# Move experiments/data/*.py → src/data/
mv evelyn_files/preprocess.py src/data/preprocessing_core.py
mv experiments/data/*.py src/data/

# 4. Move experiments/evaluation/ → src/evaluation/
mv experiments/evaluation/*.py src/evaluation/
```

**Update imports**:
```python
# Old:
from models import TabPFNModel, XGBoostModel
from evaluation import compute_metrics

# New:
from src.models import TabPFNModel, XGBoostModel
from src.evaluation import compute_metrics
```

### Phase 2: Organize Experiments

```bash
# 1. Consolidate configs
mkdir -p experiments/configs/experiments/{within_county,data_scaling,pooling}
mv experiments/config/*.yaml experiments/configs/
mv experiments/config/experiments/*.yaml experiments/configs/experiments/

# 2. Organize scripts
mkdir -p experiments/scripts/slurm
mv count_county_rows.py experiments/scripts/
mv save_counties_separately.py experiments/scripts/save_counties.py
mv experiments/scripts/slurm_*.sh experiments/scripts/slurm/

# 3. Keep only experiment-specific code in experiments/
# runners/, run_experiment.py, configs/, scripts/, analysis/
```

### Phase 3: Consolidate Notebooks

```bash
# 1. Create organized notebook structure
mkdir -p notebooks/{within_county,data_scaling,model_comparison,exploratory}

# 2. Move existing notebooks
mv notebooks/*.ipynb notebooks/exploratory/  # Current root-level notebooks
mv results_temp/*/analysis/*.ipynb notebooks/  # Analysis notebooks from results

# 3. Move cook_county notebooks
mv cook_county_analysis/notebooks/*.ipynb notebooks/data_scaling/
```

### Phase 4: Archive Old Code

```bash
# 1. Move deprecated experiment directories
mv cook_county_analysis archive/
mv prior_posterior_experiments archive/
mv evelyn_files archive/  # Keep for reference

# 2. Move old results
mkdir -p archive/old_results
mv archive/results_with_assessed_value_as_feature archive/old_results/
```

### Phase 5: Make src/ Installable

Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="tabpfn_data_scarcity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "optuna",
        # ... from requirements.txt
    ],
)
```

**Benefits**:
- Clean imports: `from src.models import TabPFNModel`
- Editable install: `pip install -e .`
- Can use anywhere: notebooks, scripts, experiments

---

## Key Decisions

### 1. **src/ vs tabpfn_data_scarcity/ package name?**

**Option A**: Keep as `src/`
```python
from src.models import TabPFNModel
```
- Pro: Clear separation (library vs experiments)
- Con: Awkward import prefix

**Option B**: Rename to package name
```python
from tabpfn_data_scarcity.models import TabPFNModel
```
- Pro: More professional, publishable
- Con: Long name, harder to type

**Recommendation**: Use `src/` for now (simple). Can refactor to package name later if publishing.

### 2. **results/ - Local or symlink?**

**Option A**: Local directory
- Pro: Everything in repo
- Con: Large files, not practical for /scratch/ data

**Option B**: Symlink to /scratch/
```bash
ln -s /scratch/users/salilg/property_tax/results results
```
- Pro: Results stay on cluster storage
- Con: Broken link on local machine

**Recommendation**: Symlink on cluster, add `results/` to .gitignore. Keep only small result summaries in repo.

### 3. **One experiments/ vs multiple?**

**Current**: `experiments/`, `cook_county_analysis/`, `prior_posterior_experiments/`

**Proposed**: Single `experiments/` with subdirectories by type
- `experiments/runners/experiment_types/` - Different experiment designs
- `experiments/configs/experiments/` - Configs organized by type
- `experiments/analysis/` - Shared analysis code

**Why**: Single source of truth, consistent structure, easier to find things.

---

## Benefits of New Structure

### 1. Clear Separation of Concerns
```
src/              # Reusable library (import anywhere)
experiments/      # Experiment scripts (run experiments)
notebooks/        # Analysis (explore results)
results/          # Outputs (organized by experiment type)
```

### 2. Easier Imports
```python
# Before (messy):
sys.path.insert(0, '../../evelyn_files')
from preprocess import Preprocess

# After (clean):
from src.data.preprocessing import Preprocess
```

### 3. Testable Code
```python
# tests/test_models.py
from src.models import TabPFNModel

def test_tabpfn_initialization():
    model = TabPFNModel()
    assert model is not None
```

### 4. Installable Package
```bash
pip install -e .  # Install in development mode

# Now from any notebook or script:
from src.models import TabPFNModel  # Just works!
```

### 5. Cleaner Git Status
```
# Before:
50+ untracked files scattered around

# After:
Clear structure, easy to ignore results/ and data/
```

---

## Migration Checklist

### Immediate (High Priority)
- [ ] Create `src/` structure
- [ ] Move `evelyn_files/preprocess.py` → `src/data/preprocessing_core.py`
- [ ] Move `experiments/models/` → `src/models/`
- [ ] Update imports in experiments/
- [ ] Test that experiments still run

### Soon (Medium Priority)
- [ ] Move root scripts → `experiments/scripts/`
- [ ] Consolidate notebooks → `notebooks/`
- [ ] Create `setup.py` for installable package
- [ ] Archive old experiment directories

### Later (Low Priority)
- [ ] Add unit tests in `tests/`
- [ ] Create proper documentation in `docs/`
- [ ] Consider renaming `src/` → `tabpfn_data_scarcity/`

---

## Example: How Code Would Look After

### Before
```python
# experiments/runners/within_county_runner.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TabPFNModel
from data import CountyDataLoader
from evaluation import compute_metrics
```

### After
```python
# experiments/runners/within_county_runner.py
from src.models import TabPFNModel
from src.data import CountyDataLoader
from src.evaluation import compute_metrics
```

**Much cleaner!**

---

## Questions to Decide

1. **src/ vs package name?** → Start with `src/`, rename later if needed
2. **Local results/ or symlink?** → Symlink to /scratch/, .gitignore
3. **Keep data/ in repo?** → No, symlink to /scratch/ or just document location
4. **When to archive cook_county_analysis/?** → After verifying data_scaling works
5. **Make pip installable now or later?** → Later (Phase 5)

---

## Recommendation

**Start with Phase 1** (Create src/ and consolidate preprocessing):
1. Move `evelyn_files/` → `src/data/preprocessing_core.py`
2. Move `experiments/models/` → `src/models/`
3. Update imports
4. Test

**Then Phase 2-3** (Organize experiments and notebooks)

**Then Phase 4** (Archive old code)

This gives you a much cleaner structure that's easier to extend with new experiments (pooling, fine-tuning, etc.)!
