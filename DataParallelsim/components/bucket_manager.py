"""Bucket manager for gradient bucketing."""

import torch.nn as nn
from typing import Dict, List, Tuple, Callable

from .bucket import GradientBucket
from ..core.config import BucketConfig

class BucketManager:
    """Manages gradient buckets and their lifecycle."""
    
    def __init__(self, config: BucketConfig):
        self.config = config
        self.buckets: Dict[int, GradientBucket] = {}
    
    def create_buckets(self, model: nn.Module) -> Dict[int, GradientBucket]:
        """Create gradient buckets from model parameters."""
        if not self.config.gradient_as_bucket_view:
            return {}
        
        capacity_bytes = self.config.capacity_mb * 1024 * 1024
        buckets = {}
        current_params = []
        current_size = 0
        bucket_id = 0
        
        # Iterate parameters in reverse order
        for param in reversed(list(model.parameters())):
            if not param.requires_grad:
                continue
            
            param_bytes = param.numel() * param.element_size()
            
            # Start new bucket if capacity exceeded and current bucket not empty
            if current_params and (current_size + param_bytes > capacity_bytes):
                buckets[bucket_id] = GradientBucket(bucket_id, current_params[:])
                bucket_id += 1
                current_params = []
                current_size = 0
            
            current_params.append(param)
            current_size += param_bytes
        
        # Add final bucket
        if current_params:
            buckets[bucket_id] = GradientBucket(bucket_id, current_params[:])
        
        self.buckets = buckets
        return buckets
    
    def register_hooks(self, bucket_ready_callback: Callable[[int], None]) -> None:
        """Register gradient hooks for all buckets."""
        def make_hook(bucket_id: int):
            def hook(grad):
                bucket_ready_callback(bucket_id)
                return grad
            return hook
        
        for bucket_id, bucket in self.buckets.items():
            bucket.register_hooks(make_hook(bucket_id))
    
    def remove_all_hooks(self) -> None:
        """Remove hooks from all buckets."""
        for bucket in self.buckets.values():
            bucket.remove_hooks()
    
    def get_bucket_info(self) -> List[Tuple[int, int, float]]:
        """Get information about all buckets: (id, params_count, size_mb)."""
        return [
            (bucket_id, len(bucket.parameters), bucket.size_mb())
            for bucket_id, bucket in self.buckets.items()
        ]