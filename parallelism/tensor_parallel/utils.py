"""
Utility Functions for Tensor Parallelism

This module provides a collection of helper functions specifically designed
to assist with tensor parallelism operations. These utilities simplify common
tasks such as retrieving tensor parallel rank and world size, checking
distributed environment status, and conditional printing.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

These utilities are generally used internally by the tensor parallelism
layers and rewriting functions, but can also be useful for debugging or
custom implementations. They abstract away direct calls to `torch.distributed`
for tensor parallelism-specific queries.

===============================================================================
"""

import torch
import torch.distributed as dist


def get_tensor_parallel_rank() -> int:
    """
    Retrieves the rank of the current process within the default distributed group.

    Note: In a pure tensor parallel setup, this would be the rank within the
    tensor parallel group. In a hybrid setup, this typically refers to the
    global rank. For specific TP group rank, use `dist.get_rank(tp_group)`.

    Returns:
        int: The global rank of the current process.
    """
    return dist.get_rank()


def get_tensor_parallel_world_size() -> int:
    """
    Retrieves the total number of processes in the default distributed group.

    Note: Similar to `get_tensor_parallel_rank`, this typically refers to the
    global world size. For specific TP group world size, use
    `dist.get_world_size(tp_group)`.

    Returns:
        int: The total number of processes in the default distributed group.
    """
    return dist.get_world_size()


def is_tensor_parallel_available() -> bool:
    """
    Checks if the distributed environment is initialized and available.

    Returns:
        bool: True if `torch.distributed` is available and initialized, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def synchronize():
    """
    Synchronizes all processes in the default distributed group.

    This acts as a barrier, ensuring that all processes reach this point
    before any proceeds further. Only performs a barrier if the distributed
    environment is initialized.
    """
    if is_tensor_parallel_available():
        dist.barrier()


def print_rank_0(message: str):
    """
    Prints a message only from the process with global rank 0.

    This is useful for avoiding duplicate log messages in a distributed setting.

    Args:
        message (str): The message to print.
    """
    if not is_tensor_parallel_available() or dist.get_rank() == 0:
        print(message)


def ensure_divisibility(numerator: int, denominator: int):
    """
    Ensures that the numerator is perfectly divisible by the denominator.

    Raises an assertion error if the division results in a remainder.
    This is commonly used in tensor parallelism to ensure that dimensions
    can be evenly sharded across ranks.

    Args:
        numerator (int): The number to be divided.
        denominator (int): The divisor.

    Raises:
        AssertionError: If `numerator` is not divisible by `denominator`.
    """
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
