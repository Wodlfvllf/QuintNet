"""
TensorParallelism package for distributed tensor parallel training.

This package provides utilities for tensor parallelism including:
- Parallel layers (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
- Model rewriting utilities
- Process group management
"""

from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .model_wrapper import apply_tensor_parallel


__all__ = [
    'ColumnParallelLinear',
    'RowParallelLinear',
    'VocabParallelEmbedding',
    'apply_tensor_parallel'
]