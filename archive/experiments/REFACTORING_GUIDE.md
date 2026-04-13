# Experiment Framework Refactoring Guide

## Overview

The experiment framework has been refactored to eliminate code duplication and provide a clean, extensible architecture for different experiment types.

### Key Changes

**Before**: Separate runner files with duplicated logic (~1,400 lines total)
- `within_county_runner.py` (554 lines)
- `cook_county_runner.py` (310 lines)
- `cross_county_runner.py` (525 lines)

**After**: Unified framework with shared base class (~800 lines total)
- `base_runner.py` (300 lines) - Shared model/training logic
- `experiment_types/` - Experiment-specific strategies
  - `data_scaling.py` (250 lines) - Replaces cook_county_runner.py
  - `within_county.py` (TBD) - Will use base_runner
  - `cross_county.py` (TBD) - Will use base_runner
- `run_experiment.py` - Unified CLI entry point

---

## New Architecture

### Base Runner (`runners/base_runner.py`)

Provides shared functionality:
- ✅ Model initialization (TabPFN, XGBoost)
- ✅ Training and prediction pipeline
- ✅ Metrics computation
- ✅ Result saving (CSV, predictions, calibration)
- ✅ Config management

### Experiment Types (`experiment_types/`)

Each experiment type implements:
- Data loading strategy
- Split generation logic
- Experiment loop
- Experiment-specific metadata

**Currently implemented:**
- `data_scaling.py` - Vary training data size (learning curves)

**To be refactored:**
- `within_county.py` - Within-county repeated k-fold CV
- `cross_county.py` - Cross-county train/test splits

**Future additions:**
- `in_context_pooling.py` - Pool data from related counties
- `fine_tuning.py` - Fine-tune TabPFN on domain data

---

## Usage

### Unified CLI: `run_experiment.py`

Single entry point for all experiments:

```bash
# Data scaling experiment (replaces cook_county_runner.py)
python experiments/run_experiment.py \
  --experiment_type data_scaling \
  --config config/experiments/data_scaling/cook_county_example.yaml

# Within-county CV (backward compatible with existing runner)
python experiments/run_experiment.py \
  --experiment_type within_county \
  --fips 1011 \
  --bin_name small \
  --k_folds 5 \
  --config config/experiments/with_preprocessing.yaml
```

### Data Scaling Experiment

**Old way** (cook_county_runner.py):
```bash
python experiments/runners/cook_county_runner.py \
  config/experiments/cook_county_with_preprocessing.yaml
```

**New way** (unified CLI):
```bash
python experiments/run_experiment.py \
  --experiment_type data_scaling \
  --config config/experiments/data_scaling/cook_county_example.yaml
```

**Config changes:**
- Add `train_sizes: [50, 100, 200, 500, 1000]`
- Add `seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Works with any county, not just Cook County!

**Benefits:**
- ✅ Can use with any county (not Cook County specific)
- ✅ Cleaner config structure
- ✅ Consistent output format
- ✅ Supports predictions and calibration saving
- ✅ ~50% less code

---

## Migration Guide

### For Data Scaling Experiments (Cook County)

1. **Create new config** in `config/experiments/data_scaling/`:
   ```yaml
   experiment:
     name: "my_experiment"

   train_sizes: [100, 500, 1000]
   seeds: [0, 1, 2, 3, 4]

   data:
     cook_county_csv: "path/to/data.csv"
   ```

2. **Run with new CLI**:
   ```bash
   python experiments/run_experiment.py \
     --experiment_type data_scaling \
     --config config/experiments/data_scaling/my_experiment.yaml
   ```

3. **Results** saved to:
   ```
   /scratch/users/salilg/property_tax/results/data_scaling/my_experiment/
   ├── results.csv
   ├── experiment.log
   └── predictions.parquet  # if enabled
   ```

### For Within-County Experiments

**No changes needed!** The old interface still works:

```bash
python experiments/runners/within_county_runner.py \
  --fips 1011 \
  --bin_name small \
  --k_folds 5 \
  --config config/experiments/with_preprocessing.yaml
```

Or use the new unified CLI (same functionality):

```bash
python experiments/run_experiment.py \
  --experiment_type within_county \
  --fips 1011 \
  --bin_name small \
  --k_folds 5 \
  --config config/experiments/with_preprocessing.yaml
```

---

## Adding New Experiment Types

### Example: In-Context Pooling

1. **Create** `experiments/experiment_types/in_context_pooling.py`:

```python
from runners.base_runner import BaseExperimentRunner

class InContextPoolingExperiment(BaseExperimentRunner):
    def run_experiment(self):
        # Load target county
        X_target, y_target = self.load_county(self.target_fips)

        # Load pool counties
        for pool_size in self.config['pool_sizes']:
            X_pool, y_pool = self.load_pool_counties(pool_size)

            # Train with pooled data as context
            for model_name in self.get_enabled_models():
                result, _, _ = self.train_and_predict(
                    model_name=model_name,
                    X_train=concat([X_pool, X_target]),  # TabPFN uses as context
                    y_train=concat([y_pool, y_target]),
                    X_test=X_test,
                    y_test=y_test
                )
                # ... collect results
```

2. **Add to** `experiment_types/__init__.py`:
```python
from .in_context_pooling import InContextPoolingExperiment
__all__ = [..., 'InContextPoolingExperiment']
```

3. **Add handler** in `run_experiment.py`:
```python
def run_in_context_pooling(config, args):
    runner = InContextPoolingExperiment(config)
    results = runner.run_experiment()
    runner.save_results(results, ...)
```

4. **Run**:
```bash
python experiments/run_experiment.py \
  --experiment_type in_context_pooling \
  --target_fips 1011 \
  --pool_sizes 5,10,20 \
  --config config/experiments/pooling_config.yaml
```

---

## File Organization

### Before
```
experiments/
├── runners/
│   ├── within_county_runner.py     # 554 lines
│   ├── cook_county_runner.py       # 310 lines (DUPLICATES within_county logic)
│   └── cross_county_runner.py      # 525 lines (DUPLICATES within_county logic)
├── config/experiments/
│   ├── cook_county_*.yaml
│   └── with_preprocessing.yaml
```

### After
```
experiments/
├── runners/
│   ├── base_runner.py              # NEW: Shared logic (300 lines)
│   ├── within_county_runner.py     # OLD: Still works (backward compat)
│   └── cross_county_runner.py      # OLD: Will be refactored
├── experiment_types/               # NEW: Experiment strategies
│   ├── __init__.py
│   ├── data_scaling.py             # Replaces cook_county_runner.py
│   ├── within_county.py            # TODO: Refactor
│   ├── cross_county.py             # TODO: Refactor
│   └── in_context_pooling.py       # FUTURE
├── run_experiment.py               # NEW: Unified CLI
├── config/experiments/
│   ├── data_scaling/               # NEW: Data scaling configs
│   │   └── cook_county_example.yaml
│   ├── with_preprocessing.yaml     # OLD: Still works
│   └── pooling/                    # FUTURE
```

---

## Benefits

### 1. No More Code Duplication
- Model initialization: **1 place** instead of 3
- Training logic: **1 place** instead of 3
- Metrics computation: **1 place** instead of 3
- Result saving: **1 place** instead of 3

### 2. Easier to Add New Experiments
- New experiment = **~100 lines** (just data strategy)
- Old way = **~300 lines** (reimplement everything)

### 3. Consistent Interface
- All experiments use same CLI
- All experiments save results the same way
- All experiments support predictions/calibration

### 4. Better Testing
- Test base_runner once → works for all experiments
- Test experiment-specific logic separately

### 5. Cleaner Configs
- Experiment type explicit in CLI (not in config)
- Config focuses on experiment parameters
- Easy to compare different experiment types

---

## Backward Compatibility

**Old scripts still work!** No breaking changes:

```bash
# Still works
python experiments/runners/within_county_runner.py --fips 1011 ...
python experiments/runners/cook_county_runner.py config.yaml

# New way (optional upgrade)
python experiments/run_experiment.py --experiment_type within_county --fips 1011 ...
python experiments/run_experiment.py --experiment_type data_scaling --config config.yaml
```

---

## Next Steps

### Phase 1: ✅ Complete
- [x] Create `base_runner.py`
- [x] Create `experiment_types/data_scaling.py`
- [x] Create unified CLI `run_experiment.py`
- [x] Test with Cook County data

### Phase 2: Refactor Existing Runners
- [ ] Refactor `within_county_runner.py` to use `base_runner`
- [ ] Refactor `cross_county_runner.py` to use `base_runner`
- [ ] Deprecate old direct runner usage

### Phase 3: New Experiment Types
- [ ] Implement `in_context_pooling.py`
- [ ] Implement `fine_tuning.py`
- [ ] Add progressive training experiments

### Phase 4: Cleanup
- [ ] Move `cook_county_analysis/` to `archive/`
- [ ] Remove duplicate files
- [ ] Update all documentation

---

## Questions?

See:
- `runners/base_runner.py` - Docstrings explain all methods
- `experiment_types/data_scaling.py` - Example implementation
- `run_experiment.py --help` - CLI usage

Or check the old runners for comparison:
- `runners/within_county_runner.py` (original implementation)
- `runners/cook_county_runner.py` (being replaced)
