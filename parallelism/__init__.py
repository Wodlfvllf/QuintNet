"""
Parallelism Strategies for Distributed Training.

This module contains implementations of:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Hybrid 3D Parallelism (DP + TP + PP)
"""

from QuintNet.parallelism.data_parallel import DataParallel, CustomDDP
from QuintNet.parallelism.tensor_parallel import TensorParallel, ColumnParallelLinear, RowParallelLinear
from QuintNet.parallelism.pipeline_parallel import PipelineParallel, PipelineParallelWrapper, PipelineTrainer
from QuintNet.parallelism.hybrid import HybridParallel, HybridParallelCoordinator

__all__ = [
    # Data Parallel
    'DataParallel',
    'CustomDDP',

    
    # Tensor Parallel
    'TensorParallel',
    'ColumnParallelLinear',
    'RowParallelLinear',
    
    # Pipeline Parallel
    'PipelineParallel',
    'PipelineParallelWrapper',
    'PipelineTrainer',
    
    # Hybrid
    'HybridParallel',
    'HybridParallelCoordinator',
]
