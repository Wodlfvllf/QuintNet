"""
Gradient Reduction Component

This module defines the `GradientReducer` class, which is responsible for
performing the collective communication operation (all-reduce) on gradients
within a `GradientBucket`.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

After gradients for a `GradientBucket` have been computed and collected,
they need to be synchronized across all participating processes in the
data parallel group. The `GradientReducer` orchestrates this `all-reduce`
operation.

Key functionalities:
-   **Flattening**: Converts the individual parameter gradients within a bucket
    into a single contiguous tensor, which is more efficient for communication.
-   **All-Reduce**: Uses the configured `DistributedBackend` to perform the
    `all-reduce` operation, summing (or otherwise reducing) the gradients
    from all processes.
-   **Averaging**: Applies the specified `ReductionStrategy` (e.g., `MEAN`)
    to the reduced gradients, typically by dividing by the world size.
-   **Unflattening**: Distributes the reduced and averaged gradients back
    to their respective `nn.Parameter` objects.

This component is crucial for ensuring that all model replicas maintain
synchronized weights during the training process.

===============================================================================
"""

import torch.distributed as dist

from .backends.base import DistributedBackend
from ..core.config import DistributedConfig, ReductionStrategy
from .bucket import GradientBucket

class GradientReducer:
    """
    Handles gradient reduction operations for `GradientBucket` instances.

    It orchestrates the flattening, all-reducing, and unflattening of gradients
    across the distributed data parallel group.
    """
    
    def __init__(self, backend: DistributedBackend, distributed_config: DistributedConfig, 
                 reduction_strategy: ReductionStrategy = ReductionStrategy.MEAN):
        """
        Initializes the GradientReducer.

        Args:
            backend (DistributedBackend): The distributed communication backend to use.
            distributed_config (DistributedConfig): Configuration for the
                distributed environment, including the process group.
            reduction_strategy (ReductionStrategy): The strategy to use for
                reducing gradients (e.g., MEAN for averaging).
        """
        self.backend = backend
        self.config = distributed_config
        self.reduction_strategy = reduction_strategy
    
    def reduce_bucket(self, bucket: GradientBucket) -> None:
        """
        Reduces gradients within a single `GradientBucket` across the
        distributed data parallel group.

        This involves flattening the gradients, performing an all-reduce,
        applying the reduction strategy (e.g., averaging), and then
        unflattening the gradients back to the parameters.

        Args:
            bucket (GradientBucket): The bucket containing gradients to be reduced.
        """
        # If no gradients are present in the bucket, mark it as not ready and return.
        if not bucket.has_gradients():
            bucket.is_ready = False
            return
        
        try:
            # 1. Flatten gradients from all parameters in the bucket into a single tensor.
            flattened_tensor, device = bucket.flatten_gradients()
            
            # 2. Perform the all-reduce operation using the configured backend.
            if self.backend.is_initialized():
                self.backend.all_reduce_tensor(
                    flattened_tensor, 
                    dist.ReduceOp.SUM, # Always sum during all-reduce, then divide for mean
                    self.config.process_group
                )
                
                # 3. Apply the reduction strategy (e.g., divide by world size for mean).
                world_size = self.backend.get_world_size(self.config.process_group)
                if self.reduction_strategy == ReductionStrategy.MEAN and world_size > 1:
                    flattened_tensor.div_(world_size)
            
            # 4. Unflatten the reduced tensor back into the individual parameter gradients.
            bucket.unflatten_gradients(flattened_tensor)
            
        except Exception as e:
            print(f"Error reducing bucket {bucket.bucket_id}: {e}")
        finally:
            # Reset the bucket's ready state regardless of success or failure.
            bucket.is_ready = False
