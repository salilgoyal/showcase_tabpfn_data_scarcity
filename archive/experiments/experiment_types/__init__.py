"""
Experiment type modules for different experimental designs.

Each module implements a specific type of experiment by inheriting from
BaseExperimentRunner and defining its data splitting and evaluation strategy.

Available experiment types:
- data_scaling: Vary training data size (e.g., Cook County subsampling)
- within_county: Within-county repeated k-fold cross-validation
- cross_county: Cross-county train/test splits

Future experiment types:
- in_context_pooling: Pool data from related counties for in-context learning
- fine_tuning: Fine-tune TabPFN on domain-specific data
"""

from .data_scaling import DataScalingExperiment

__all__ = ['DataScalingExperiment']
