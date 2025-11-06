"""
TensorParallelism package for distributed tensor parallel training.

This package provides utilities for tensor parallelism including:
- Communication operations (All_Gather, All_Reduce, ReduceScatter)
- Parallel layers (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
- Model rewriting utilities
- Process group management
"""

from QuintNet.core.communication import All_Gather, All_Reduce, ReduceScatter
from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .rewrite import apply_tensor_parallel


__all__ = [
    'All_Gather',
    'All_Reduce', 
    'ReduceScatter',
    'ColumnParallelLinear',
    'RowParallelLinear',
    'VocabParallelEmbedding',
    'apply_tensor_parallel'
]