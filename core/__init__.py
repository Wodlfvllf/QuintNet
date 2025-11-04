"""
Core utilities for QuintNet distributed training.

This module contains fundamental abstractions for:
- Process mesh management
- Distributed backend initialization
- Process group management
- Communication primitives
"""

from QuintNet.core.mesh import MeshGenerator, init_mesh
from QuintNet.core.distributed import setup_distributed, cleanup_distributed
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.core.communication import (
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
