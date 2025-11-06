"""Main CustomDDP implementation."""

import torch.nn as nn
from typing import Optional, List, Tuple

from QuintNet.parallelism.data_parallel.backends.base import DistributedBackend
from QuintNet.parallelism.data_parallel.backends.torch_backend import TorchDistributedBackend
from QuintNet.parallelism.data_parallel.components.bucket_manager import BucketManager
from QuintNet.parallelism.data_parallel.components.gradient_reducer import GradientReducer
from QuintNet.parallelism.data_parallel.components.parameter_broadcaster import ParameterBroadcaster
from QuintNet.parallelism.data_parallel.core.config import DistributedConfig, BucketConfig, ReductionStrategy

class CustomDDP(nn.Module):
    """
    Modular custom DDP implementation with pluggable components.
    """
    
    def __init__(
        self,
        model: nn.Module,
        distributed_config: Optional[DistributedConfig] = None,
        bucket_config: Optional[BucketConfig] = None,
        backend: Optional[DistributedBackend] = None,
        reduction_strategy: ReductionStrategy = ReductionStrategy.MEAN,
    ):
        super().__init__()
        
        self.model = model
        self.distributed_config = distributed_config or DistributedConfig()
        self.bucket_config = bucket_config or BucketConfig()
        self.backend = backend or TorchDistributedBackend()
        
        # Initialize components
        self.bucket_manager = BucketManager(self.bucket_config)
        self.gradient_reducer = GradientReducer(self.backend, self.distributed_config, reduction_strategy)
        self.parameter_broadcaster = ParameterBroadcaster(self.backend, self.distributed_config)
        
        # Setup
        self._setup()
    
    def _setup(self) -> None:
        """Initialize the DDP wrapper."""
        # Create buckets
        self.buckets = self.bucket_manager.create_buckets(self.model)
        
        # Broadcast parameters
        self.parameter_broadcaster.broadcast_parameters(self.model)
        
        # Register hooks if using bucketed gradients
        if self.buckets:
            self.bucket_manager.register_hooks(self._on_bucket_ready)
    
    def _on_bucket_ready(self, bucket_id: int) -> None:
        """Callback when a bucket is ready for reduction."""
        bucket = self.buckets[bucket_id]
        bucket.is_ready = True
        
        if bucket.has_gradients():
            self.gradient_reducer.reduce_bucket(bucket)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def remove_hooks(self) -> None:
        """Remove all gradient hooks."""
        self.bucket_manager.remove_all_hooks()
    
    def get_bucket_info(self) -> List[Tuple[int, int, float]]:
        """Get information about gradient buckets."""
        return self.bucket_manager.get_bucket_info()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass