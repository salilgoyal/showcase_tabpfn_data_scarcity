# Outdated Notebooks (archived 2025-12-30)

These notebooks have outdated imports that don't match the current codebase structure:
- cook_county_archive.ipynb - Uses old `src.data_utils` and `src.model_runners` imports

If you want to restore these, update the imports to:
- `from src.data import load_and_prepare_data`
- `from src.models import TabPFNModel, XGBoostModel`
