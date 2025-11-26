"""
Core utilities for QuintNet distributed training.

This module contains fundamental abstractions for:
- Process mesh management
- Distributed backend initialization
- Process group management
- Communication primitives
"""

from .mesh import MeshGenerator
from .distributed import setup_distributed, cleanup_distributed
from .process_groups import ProcessGroupManager, init_process_groups
from .config import load_config
from .communication import (
    All_Reduce,
    All_Gather,
    ReduceScatter,
    Send,
    Recv,
    pipeline_communicate,
    bidirectional_pipeline_communicate,
)

__all__ = [
    'MeshGenerator',
    'init_process_groups',
    'setup_distributed',
    'cleanup_distributed',
    'ProcessGroupManager',
    'load_config',
    'All_Reduce',
    'All_Gather',
    'ReduceScatter',
    'Send',
    'Recv',
    'pipeline_communicate',
    'bidirectional_pipeline_communicate',
]
