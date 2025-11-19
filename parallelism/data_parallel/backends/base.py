"""
Abstract Base Class for Distributed Backends

This module defines the abstract interface for distributed communication
backends used within the `CustomDDP` implementation. By abstracting the
communication primitives, `CustomDDP` can be made agnostic to the specific
distributed library (e.g., PyTorch's `torch.distributed`, MPI, etc.).

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The `DistributedBackend` class specifies the fundamental communication
operations required for data parallelism, such as broadcasting tensors
and performing all-reduce operations. Any concrete implementation of this
abstract class must provide these methods.

This design promotes modularity and extensibility, allowing different
distributed communication libraries to be plugged into the `CustomDDP`
framework without modifying its core logic.

===============================================================================
"""

import torch
import torch.distributed as dist
from typing import Optional
from abc import ABC, abstractmethod

class DistributedBackend(ABC):
    """
    Abstract interface for distributed communication backends.

    This class defines the essential communication primitives that any
    distributed backend must implement to be compatible with `CustomDDP`.
    """
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Checks if the distributed backend is currently initialized.

        Returns:
            bool: True if the backend is initialized, False otherwise.
        """
        pass
    
    @abstractmethod
    def broadcast_tensor(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None) -> None:
        """
        Broadcasts a tensor from a source rank to all other ranks in a group.

        Args:
            tensor (torch.Tensor): The tensor to be broadcasted. On the source
                rank, this tensor contains the data to send. On other ranks,
                it will be overwritten with the received data.
            src (int): The source rank from which to broadcast.
            group (Optional[dist.ProcessGroup]): The process group over which
                to perform the broadcast. If None, the default group is used.
        """
        pass
    
    @abstractmethod
    def all_reduce_tensor(self, tensor: torch.Tensor, op: dist.ReduceOp, group: Optional[dist.ProcessGroup] = None) -> None:
        """
        Performs an all-reduce operation on a tensor across all ranks in a group.

        The result of the reduction (e.g., sum, mean) is available on all ranks.

        Args:
            tensor (torch.Tensor): The tensor to be all-reduced. It will be
                overwritten with the result of the reduction.
            op (dist.ReduceOp): The reduction operation to apply (e.g., `dist.ReduceOp.SUM`).
            group (Optional[dist.ProcessGroup]): The process group over which
                to perform the all-reduce. If None, the default group is used.
        """
        pass
    
    @abstractmethod
    def get_world_size(self, group: Optional[dist.ProcessGroup] = None) -> int:
        """
        Retrieves the world size (number of processes) for a given process group.

        Args:
            group (Optional[dist.ProcessGroup]): The process group. If None,
                the world size of the default group is returned.

        Returns:
            int: The world size of the specified group.
        """
        pass
