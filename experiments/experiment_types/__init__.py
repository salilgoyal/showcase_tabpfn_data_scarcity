"""
Experiment type modules for different experimental designs.

Each module implements a specific type of experiment by inheriting from
BaseExperimentRunner and defining its data splitting and evaluation strategy.

Available experiment types:
- data_scaling: Vary training data size (e.g., Cook County subsampling)
- within_county: Within-county repeated k-fold cross-validation
- cross_county: Cross-county train/test splits
- finetuning: Fine-tune TabPFN on large-scale pooled data
- per_county_scaling: Per-county learning curves for tiny/small counties

Future experiment types:
- in_context_pooling: Pool data from related counties for in-context learning
"""

from .base import BaseExperimentRunner, ExperimentMetadata
from .data_scaling import DataScalingExperiment
from .cross_county import CrossCountyExperiment
from .finetuning import FinetuningExperiment
from .per_county_scaling import PerCountyScalingExperiment
from .geo_pooling import GeoPoolingExperiment
from .global_finetuning import GlobalFinetuningExperiment
# from .within_county import WithinCountyExperiment  # Commented out - needs migration to CleanedDataLoader

__all__ = [
    'BaseExperimentRunner',
    'ExperimentMetadata',
    'DataScalingExperiment',
    'CrossCountyExperiment',
    'FinetuningExperiment',
    'PerCountyScalingExperiment',
    'GeoPoolingExperiment',
    'GlobalFinetuningExperiment',
    # 'WithinCountyExperiment',  # Commented out - needs migration
]
