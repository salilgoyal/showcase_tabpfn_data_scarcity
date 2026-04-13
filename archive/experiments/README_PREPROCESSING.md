# Preprocessing Options - Quick Reference

The experiments framework now supports two preprocessing pipelines:

## Option 1: Original Preprocessing (Default)
- Simple median imputation
- No transformations
- Drops admin columns only
- ~150 features per county

## Option 2: Evelyn's Preprocessing (Recommended)
- Winsorization (outlier handling)
- **Log transformation of target**
- Feature normalization
- Temporal feature generation
- ~21 features (minimal) or ~110 features (full)

## Enable Evelyn's Preprocessing

Edit `config/base_config.yaml`:

```yaml
preprocessing:
  use_evelyn_preprocessing: true   # Toggle here
  include_property_chars: false    # Minimal vs full features
```

## Important Notes

✅ **Metrics are on original scale** - Log transformation is handled automatically
✅ **Backward compatible** - Default behavior unchanged
✅ **No code changes needed** - Just update config

## Documentation

- **Quick Start**: `docs/QUICK_START.md` - Get started in 2 minutes
- **Full Guide**: `docs/PREPROCESSING_GUIDE.md` - Complete reference
- **Technical Details**: `docs/IMPLEMENTATION_SUMMARY.md` - Implementation notes
- **Changes**: `docs/CHANGES.md` - What was modified

## Questions?

See the documentation above or compare with `cook_county_analysis/docs/` for examples.
