"""
Centralized path utilities for the project.

Provides PROJECT_ROOT and helper functions for resolving paths consistently.
"""

from pathlib import Path


# Project root is two levels up from this file (src/utils/paths.py -> src/ -> project/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_data_path(path: str) -> Path:
    """
    Resolve a data path, handling both absolute and relative paths.

    Relative paths are resolved relative to PROJECT_ROOT.

    Args:
        path: Path to resolve (absolute or relative)

    Returns:
        Resolved absolute Path object

    Examples:
        >>> resolve_data_path("/absolute/path/data.csv")
        Path("/absolute/path/data.csv")

        >>> resolve_data_path("data/county_csvs/fips_123.csv")
        Path("/project/root/data/county_csvs/fips_123.csv")
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / path


def get_config_dir() -> Path:
    """Get the experiments config directory."""
    return PROJECT_ROOT / "experiments" / "configs"


def get_results_dir() -> Path:
    """Get the default results directory."""
    return PROJECT_ROOT / "results"


def get_logs_dir() -> Path:
    """Get the default logs directory."""
    return PROJECT_ROOT / "logs"
