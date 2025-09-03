"""Abstract base class for distributed backends."""

import torch
import torch.distributed as dist
from typing import Optional
from abc import ABC, abstractmethod

class DistributedBackend(ABC):
    """Abstract interface for distributed communication."""
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if distributed backend is initialized."""
        pass
    
    @abstractmethod
    def broadcast_tensor(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None) -> None:
        """Broadcast tensor from source rank to all ranks."""
        pass
    
    @abstractmethod
    def all_reduce_tensor(self, tensor: torch.Tensor, op: dist.ReduceOp, group: Optional[dist.ProcessGroup] = None) -> None:
        """All-reduce tensor across all ranks."""
        pass
    
    @abstractmethod
    def get_world_size(self, group: Optional[dist.ProcessGroup] = None) -> int:
        """Get world size for the process group."""
        pass