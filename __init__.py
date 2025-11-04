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
from QuintNet.core.mesh import MeshGenerator, init_mesh
from QuintNet.core.distributed import setup_distributed, cleanup_distributed

# Parallelism strategies
from QuintNet.parallelism import (
    DataParallel,
    TensorParallel,
    PipelineParallel,
    HybridParallel
)

__all__ = [
    # Core
    'MeshGenerator',
    'init_mesh',
    'setup_distributed',
    'cleanup_distributed',
    
    # Parallelism
    'DataParallel',
    'TensorParallel',
    'PipelineParallel',
    'HybridParallel',
]
