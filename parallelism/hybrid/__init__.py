"""
Hybrid 3D Parallelism (DP + TP + PP).

This module contains:
- Coordinator for all three parallelism strategies
- Strategy selection and configuration
- Unified training interface

Migration Source: Logic scattered across Train_DP_TP_PP/train_dp_tp_pp.py
"""

from QuintNet.parallelism.hybrid.coordinator import HybridParallelCoordinator
from QuintNet.parallelism.hybrid.strategy import ParallelismStrategy, HybridParallel

__all__ = [
    'HybridParallelCoordinator',
    'ParallelismStrategy',
    'HybridParallel',
]
