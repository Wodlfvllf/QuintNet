"""
PyTorch Distributed Backend Implementation

This module provides a concrete implementation of the `DistributedBackend`
abstract class, utilizing PyTorch's `torch.distributed` package for
distributed communication primitives.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The `TorchDistributedBackend` acts as a bridge between the abstract
`DistributedBackend` interface and PyTorch's native distributed functionalities.
It wraps `torch.distributed` calls for `is_initialized`, `broadcast`,
`all_reduce`, and `get_world_size`, ensuring that `DataParallel` can
seamlessly integrate with PyTorch's distributed environment.

This separation allows for potential future extensions where other distributed
communication libraries (e.g., custom MPI wrappers) could be used by simply
implementing the `DistributedBackend` interface.

===============================================================================
"""

import torch
import torch.distributed as dist
from typing import Optional

from .base import DistributedBackend

class TorchDistributedBackend(DistributedBackend):
    """
    PyTorch distributed backend implementation.

    This class provides concrete implementations for the abstract methods
    defined in `DistributedBackend`, using `torch.distributed` for all
    communication operations.
    """
    
    def is_initialized(self) -> bool:
        """
        Checks if the `torch.distributed` backend is available and initialized.

        Returns:
            bool: True if `torch.distributed` is ready for use, False otherwise.
        """
        return dist.is_available() and dist.is_initialized()
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None) -> None:
        """
        Broadcasts a tensor from a source rank to all other ranks using `torch.distributed.broadcast`.

        Args:
            tensor (torch.Tensor): The tensor to be broadcasted.
            src (int): The source rank.
            group (Optional[dist.ProcessGroup]): The process group.
        """
        if self.is_initialized():
            rank = dist.get_rank()
            print(f"[Rank {rank}] TorchDistributedBackend: START broadcast from {src}", flush=True)
            dist.broadcast(tensor, src=src, group=group)
            print(f"[Rank {rank}] TorchDistributedBackend: END broadcast", flush=True)
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: dist.ReduceOp, group: Optional[dist.ProcessGroup] = None) -> None:
        """
        Performs an all-reduce operation on a tensor using `torch.distributed.all_reduce`.

        Args:
            tensor (torch.Tensor): The tensor to be all-reduced.
            op (dist.ReduceOp): The reduction operation.
            group (Optional[dist.ProcessGroup]): The process group.
        """
        if self.is_initialized():
            rank = dist.get_rank()
            print(f"[Rank {rank}] TorchDistributedBackend: START all_reduce", flush=True)
            dist.all_reduce(tensor, op=op, group=group)
            print(f"[Rank {rank}] TorchDistributedBackend: END all_reduce", flush=True)
    
    def get_world_size(self, group: Optional[dist.ProcessGroup] = None) -> int:
        """
        Retrieves the world size for a given process group using `torch.distributed.get_world_size`.

        Args:
            group (Optional[dist.ProcessGroup]): The process group.

        Returns:
            int: The world size. Returns 1 if `torch.distributed` is not initialized.
        """
        if self.is_initialized():
            return dist.get_world_size(group) if group else dist.get_world_size()
        return 1 # Default to 1 if distributed is not initialized (e.g., single-GPU run)
