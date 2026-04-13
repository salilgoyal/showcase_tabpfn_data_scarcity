# Obsolete Files (After Hybrid Preprocessing Migration)

## Files That Can Be Archived

These files are no longer needed with the new hybrid preprocessing architecture:

### 1. Old Preprocessing (in `src/data/`)

| File | Status | Reason |
|------|--------|--------|
| `src/data/preprocessing.py` | **Partially obsolete** | The `Preprocess` class is replaced by Phase 1 + Phase 2 split. Individual functions may still be useful for reference. |

**Decision**: Keep for now as reference, but mark as deprecated in comments. Eventually can extract useful functions.

### 2. Investigative Scripts

| File/Directory | Status | Reason |
|----------------|--------|--------|
| `investigative_scripts/feature_analysis/analyze_preprocessed_features.py` | **Obsolete** | This script preprocessed each county individually then pooled - replaced by pooled-then-preprocess approach |
| `investigative_scripts/feature_analysis/` (entire dir) | **Review needed** | May contain analysis code that's still useful for understanding features |

**Decision**: Move to `archive/` if no longer actively used.

### 3. Old Experiment Configs

| File | Status | Reason |
|------|--------|--------|
| `experiments/configs/cross_county/small_in_context_10k*.yaml` | **Outdated** | These configs use old preprocessing approach (per-county) |
| Any config with `preprocessing.features` and `preprocessing.steps` embedded | **Outdated** | Should be migrated to point to `cleaned_data_path` |

**Decision**: Keep old configs for historical reference, but clearly label as "v1" (old approach).

### 4. Old Experiment Types (Not Yet Migrated)

| File | Status | Reason |
|------|--------|--------|
| `experiments/experiment_types/within_county.py` | **Needs migration** | Still uses old `CountyDataLoader` with per-county preprocessing |
| `experiments/experiment_types/data_scaling.py` | **Needs migration** | Still uses old preprocessing approach |
| `experiments/experiment_types/finetuning.py` | **Needs migration** | May or may not be compatible with new approach |

**Decision**: These work but should be migrated to use `CleanedDataLoader`. Until then, they're "deprecated but functional".

## Definitely Obsolete (Can Delete or Archive)

### Scripts/Files No Longer Used

```
# Analysis scripts for old preprocessing
investigative_scripts/feature_analysis/analyze_preprocessed_features.py
  → Replaced by: preprocessing/scripts/clean_pooled_data.py

# Any scripts that do per-county preprocessing then pooling
# (these had the leakage issue)
```

## Migration Priority

### High Priority (Should Migrate Soon)
1. `within_county.py` - Commonly used experiment type
2. Old cross-county configs - Update to use `cross_county_v2.yaml` pattern

### Medium Priority
3. `data_scaling.py` - Less frequently used but valuable
4. Analysis notebooks that use old preprocessing

### Low Priority
5. `finetuning.py` - Can work independently
6. Investigative scripts - Only if actively using

## Backward Compatibility

**Current Status**: No backward compatibility maintained.

- Old experiment types (within_county, data_scaling) still work because `src/data/preprocessing.py` still exists
- But they use the old (leaky) preprocessing approach
- New experiments should always use `CleanedDataLoader`

## Recommended Cleanup Steps

1. **Immediate**:
   ```bash
   # Move obviously obsolete scripts
   mkdir -p archive/investigative_scripts/
   mv investigative_scripts/feature_analysis/analyze_preprocessed_features.py archive/investigative_scripts/

   # Rename old configs to indicate version
   mv experiments/configs/cross_county/small_in_context_10k.yaml \
      experiments/configs/cross_county/small_in_context_10k_v1_old.yaml
   ```

2. **Soon** (within 1-2 weeks):
   - Migrate `within_county.py` to use `CleanedDataLoader`
   - Create v2 configs for all active experiments
   - Add deprecation warnings to old code

3. **Eventually** (1-2 months):
   - Delete or fully archive `src/data/preprocessing.py`
   - Remove all v1 configs
   - Archive unmigrated experiment types

## How to Identify if a File is Obsolete

**Signs a file is obsolete**:
- ✅ Uses `CountyDataLoader` with `preprocessing_config`
- ✅ Preprocesses each county separately then pools
- ✅ Applies normalization/winsorization before train/test split
- ✅ Config has large `preprocessing:` sections with both Phase 1 and Phase 2 steps

**Signs a file is up-to-date**:
- ✅ Uses `CleanedDataLoader`
- ✅ Points to `cleaned_data_path` in config
- ✅ Only has `phase2_steps` in preprocessing config
- ✅ Applies Phase 2 per train/test split

## Questions Before Deleting

Before deleting any file, ask:
1. Is there active analysis/research depending on it?
2. Does it contain unique logic not replicated elsewhere?
3. Is it referenced in papers/notebooks?
4. Could it be useful as a comparison (old vs new approach)?

If "yes" to any → move to `archive/` with explanation
If "no" to all → safe to delete
