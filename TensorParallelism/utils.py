"""
Utility functions for tensor parallelism.

This module provides helper functions and utilities for tensor parallel operations.
"""

import torch
import torch.distributed as dist


def get_tensor_parallel_rank():
    """Get the current tensor parallel rank"""
    return dist.get_rank()


def get_tensor_parallel_world_size():
    """Get the tensor parallel world size"""
    return dist.get_world_size()


def is_tensor_parallel_available():
    """Check if tensor parallelism is available (distributed is initialized)"""
    return dist.is_available() and dist.is_initialized()


def synchronize():
    """Synchronize all processes"""
    if is_tensor_parallel_available():
        dist.barrier()


def print_rank_0(message):
    """Print message only on rank 0"""
    if not is_tensor_parallel_available() or dist.get_rank() == 0:
        print(message)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator"""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"