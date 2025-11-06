"""Gradient reduction component."""

import torch.distributed as dist

from QuintNet.parallelism.data_parallel.backends.base import DistributedBackend
from QuintNet.parallelism.data_parallel.core.config import DistributedConfig, ReductionStrategy
from QuintNet.parallelism.data_parallel.components.bucket import GradientBucket

class GradientReducer:
    """Handles gradient reduction operations."""
    
    def __init__(self, backend: DistributedBackend, distributed_config: DistributedConfig, 
                 reduction_strategy: ReductionStrategy = ReductionStrategy.MEAN):
        self.backend = backend
        self.config = distributed_config
        self.reduction_strategy = reduction_strategy
    
    def reduce_bucket(self, bucket: GradientBucket) -> None:
        """Reduce gradients in a single bucket."""
        if not bucket.has_gradients():
            bucket.is_ready = False
            return
        
        try:
            # Flatten gradients
            flattened_tensor, device = bucket.flatten_gradients()
            
            # Perform reduction
            if self.backend.is_initialized():
                self.backend.all_reduce_tensor(
                    flattened_tensor, 
                    dist.ReduceOp.SUM, 
                    self.config.process_group
                )
                
                # Apply reduction strategy
                world_size = self.backend.get_world_size(self.config.process_group)
                if self.reduction_strategy == ReductionStrategy.MEAN and world_size > 1:
                    flattened_tensor.div_(world_size)
            
            # Unflatten back to parameters
            bucket.unflatten_gradients(flattened_tensor)
            
        except Exception as e:
            print(f"Error reducing bucket {bucket.bucket_id}: {e}")
        finally:
            bucket.is_ready = False