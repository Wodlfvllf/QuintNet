"""
Core utilities for QuintNet distributed training.

This module contains fundamental abstractions for:
- Process mesh management
- Distributed backend initialization
- Process group management
- Communication primitives
"""

from .mesh import MeshGenerator, init_mesh
from .distributed import setup_distributed, cleanup_distributed
from .process_groups import ProcessGroupManager
from .communication import (
    all_reduce,
    all_gather,
    reduce_scatter,
    broadcast,
    send,
    recv
)

__all__ = [
    'MeshGenerator',
    'init_mesh',
    'setup_distributed',
    'cleanup_distributed',
    'ProcessGroupManager',
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'broadcast',
    'send',
    'recv',
]
