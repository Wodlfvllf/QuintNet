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
    all_reduce,
    all_gather,
    reduce_scatter,
    broadcast,
    send,
    recv
)

__all__ = [
    'MeshGenerator',
    'init_process_groups',
    'setup_distributed',
    'cleanup_distributed',
    'ProcessGroupManager',
    'load_config',
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'broadcast',
    'send',
    'recv',
]
