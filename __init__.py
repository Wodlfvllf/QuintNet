"""
QuintNet - Towards 5D Parallelism for Scalable Deep Learning

A comprehensive library for distributed deep learning with support for:
- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Hybrid 3D Parallelism (DP + TP + PP)

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "QuintNet Team"

# Core imports
from .strategy import (
    BaseStrategy,
    DataParallelStrategy,
    TensorParallelStrategy,
    PipelineParallelStrategy,
    DataTensorParallelStrategy,
    DataPipelineParallelStrategy,
    TensorPipelineParallelStrategy,
    Hybrid3DStrategy,
    get_strategy,
)
from .core import init_process_groups, setup_distributed, cleanup_distributed

# Parallelism strategies
from .parallelism import (
    DataParallel,
    TensorParallel,
    PipelineParallelWrapper,
    PipelineTrainer,
    PipelineDataLoader,
    # HybridParallel
)

__all__ = [
    # Core
    'init_process_groups',
    'setup_distributed',
    'cleanup_distributed',
    'get_strategy',
    
    # Parallelism
    'DataParallel',
    'TensorParallel',
    'PipelineParallelWrapper',
    'PipelineTrainer',
    'PipelineDataLoader',
    # 'HybridParallel',
]
