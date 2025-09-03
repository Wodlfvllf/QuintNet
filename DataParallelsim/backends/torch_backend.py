"""PyTorch distributed backend implementation."""

import torch
import torch.distributed as dist
from typing import Optional

from .base import DistributedBackend

class TorchDistributedBackend(DistributedBackend):
    """PyTorch distributed backend implementation."""
    
    def is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None) -> None:
        if self.is_initialized():
            dist.broadcast(tensor, src=src, group=group)
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: dist.ReduceOp, group: Optional[dist.ProcessGroup] = None) -> None:
        if self.is_initialized():
            dist.all_reduce(tensor, op=op, group=group)
    
    def get_world_size(self, group: Optional[dist.ProcessGroup] = None) -> int:
        if self.is_initialized():
            return dist.get_world_size(group) if group else dist.get_world_size()
        return 1