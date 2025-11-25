"""
Parallelism Strategies for Distributed Training.

This module contains implementations of:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Hybrid 3D Parallelism (DP + TP + PP)
"""

from .data_parallel import CustomDDP as DataParallel, CustomDDP
from .tensor_parallel import apply_tensor_parallel as TensorParallel, ColumnParallelLinear, RowParallelLinear
from .pipeline_parallel import PipelineParallelWrapper as PipelineParallel, PipelineParallelWrapper, PipelineTrainer
# from .hybrid import HybridParallel, HybridParallelCoordinator # Hybrid module missing

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
    # 'HybridParallel',
    # 'HybridParallelCoordinator',
]
