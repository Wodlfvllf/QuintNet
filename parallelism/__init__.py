"""
Parallelism Strategies for Distributed Training.

This module contains implementations of:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Hybrid 3D Parallelism (DP + TP + PP)
"""

from .data_parallel import DataParallel
from .data_parallel.components import GradientBucket, BucketManager, GradientReducer, ParameterBroadcaster
from .tensor_parallel import apply_tensor_parallel as TensorParallel, ColumnParallelLinear, RowParallelLinear
from .pipeline_parallel import PipelineParallelWrapper as PipelineParallel, PipelineParallelWrapper, PipelineTrainer

__all__ = [
    # Data Parallel
    'DataParallel',
    'GradientBucket',
    'BucketManager',
    'GradientReducer',
    'ParameterBroadcaster',
    
    # Tensor Parallel
    'TensorParallel',
    'ColumnParallelLinear',
    'RowParallelLinear',
    
    # Pipeline Parallel
    'PipelineParallel',
    'PipelineParallelWrapper',
    'PipelineTrainer',
]
