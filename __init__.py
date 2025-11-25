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
from .core.mesh import MeshGenerator
from .core.process_groups import init_process_groups
from .core.distributed import setup_distributed, cleanup_distributed

# Parallelism strategies
from .parallelism import (
    DataParallel,
    TensorParallel,
    PipelineParallel,
    # HybridParallel
)

__all__ = [
    # Core
    'MeshGenerator',
    'init_process_groups',
    'setup_distributed',
    'cleanup_distributed',
    
    # Parallelism
    'DataParallel',
    'TensorParallel',
    'PipelineParallel',
    # 'HybridParallel',
]
