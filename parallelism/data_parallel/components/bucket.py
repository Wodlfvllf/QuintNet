"""Gradient bucket implementation."""

import torch
import torch.nn as nn
from typing import List, Tuple, Callable

class GradientBucket:
    """Manages a single gradient bucket."""
    
    def __init__(self, bucket_id: int, parameters: List[nn.Parameter]):
        self.bucket_id = bucket_id
        self.parameters = parameters
        self.is_ready = False
        self._hook_handles = []
    
    def size_bytes(self) -> int:
        """Calculate total size of parameters in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters)
    
    def size_mb(self) -> float:
        """Calculate total size of parameters in MB."""
        return self.size_bytes() / (1024 ** 2)
    
    def has_gradients(self) -> bool:
        """Check if all parameters have gradients."""
        return all(p.grad is not None for p in self.parameters if p.requires_grad)
    
    def flatten_gradients(self) -> Tuple[torch.Tensor, torch.device]:
        """Flatten all gradients in the bucket."""
        flat_grads = []
        devices = []
        
        for p in self.parameters:
            if p.grad is not None:
                flat = p.grad.detach().view(-1)
                flat_grads.append(flat)
                devices.append(p.grad.device)
        
        if len(flat_grads) == 0:
            raise ValueError(f"No gradients found in bucket {self.bucket_id}")
        
        device = devices[0]
        flat_grads = [g.to(device) for g in flat_grads]
        return torch.cat(flat_grads), device
    
    def unflatten_gradients(self, flattened_tensor: torch.Tensor) -> None:
        """Unflatten tensor back to parameter gradients."""
        offset = 0
        for p in self.parameters:
            if p.grad is None:
                continue
            numel = p.grad.numel()
            chunk = flattened_tensor[offset:offset + numel]
            p.grad.data.copy_(chunk.view_as(p.grad).to(p.grad.device))
            offset += numel
    
    def register_hooks(self, hook_fn: Callable) -> None:
        """Register gradient hooks for all parameters in the bucket."""
        for p in self.parameters:
            if p.requires_grad:
                handle = p.register_hook(hook_fn)
                self._hook_handles.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []