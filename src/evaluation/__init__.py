"""Evaluation utilities for experiments."""

from .metrics import compute_metrics, compute_relative_performance
from .aggregation import ResultsAggregator

__all__ = [
    'compute_metrics',
    'compute_relative_performance',
    'ResultsAggregator',
]
