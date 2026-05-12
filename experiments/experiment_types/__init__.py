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
from .geo_pooling import GeoPoolingExperiment
from .global_finetuning import GlobalFinetuningExperiment
from .single_county_scaling import SingleCountyScalingExperiment

__all__ = [
    'BaseExperimentRunner',
    'ExperimentMetadata',
    'GeoPoolingExperiment',
    'GlobalFinetuningExperiment',
    'SingleCountyScalingExperiment',
]
