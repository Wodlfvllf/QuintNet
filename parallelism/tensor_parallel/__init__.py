"""
Tensor Parallelism Implementation.

This module contains:
- Parallel layers (Column, Row)
- Communication operations (All-Gather, Reduce-Scatter)
- Model transformation utilities

Migration Source: QuintNet/TensorParallelism/
"""

from QuintNet.parallelism.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallel
)
from QuintNet.parallelism.tensor_parallel.operations import (
    all_gather_tensor,
    reduce_scatter_tensor,
    all_reduce_tensor
)
from QuintNet.parallelism.tensor_parallel.model_wrapper import apply_tensor_parallel
from QuintNet.parallelism.tensor_parallel.process_group import ProcessGroupManager as TPProcessGroupManager

__all__ = [
    # Layers
    'ColumnParallelLinear',
    'RowParallelLinear',
    'TensorParallel',
    
    # Operations
    'all_gather_tensor',
    'reduce_scatter_tensor',
    'all_reduce_tensor',
    
    # Utilities
    'apply_tensor_parallel',
    'TPProcessGroupManager',
]
