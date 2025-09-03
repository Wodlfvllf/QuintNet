"""Local (non-distributed) backend for testing."""

import torch
import torch.distributed as dist
from typing import Optional

from .base import DistributedBackend

class LocalBackend(DistributedBackend):
    """Local (non-distributed) backend for testing."""
    
    def is_initialized(self) -> bool:
        return False
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None) -> None:
        pass  # No-op for local backend
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: dist.ReduceOp, group: Optional[dist.ProcessGroup] = None) -> None:
        pass  # No-op for local backend
    
    def get_world_size(self, group: Optional[dist.ProcessGroup] = None) -> int:
        return 1