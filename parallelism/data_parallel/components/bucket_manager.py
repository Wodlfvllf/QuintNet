"""
Bucket Manager for Gradient Bucketing

This module contains the `BucketManager` class, which is responsible for
grouping model parameters into "buckets" for efficient gradient communication
in a Distributed Data Parallel (DDP) setup.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

In DDP, after the backward pass, gradients from each GPU replica need to be
exchanged and averaged. Performing an `all-reduce` operation for each
individual parameter can be inefficient due to COMMUNICATION OVERHEAD (latency).

Gradient bucketing addresses this by grouping multiple small gradients into
larger "buckets." When all gradients within a bucket are ready, a single
`all-reduce` operation is performed on that larger bucket, significantly
reducing the number of communication calls.

The `BucketManager`:
-   Iterates through model parameters (typically in reverse order of definition)
    to fill buckets up to a specified capacity.
-   Registers hooks on the gradients of these parameters.
-   When a gradient is computed, it signals the `BucketManager`, which in turn
    calls a `bucket_ready_callback` when a full bucket of gradients is available
    for all-reduction.

This mechanism is crucial for optimizing communication in DDP, especially
with models that have many small parameters.

===============================================================================
"""

import torch.nn as nn
from typing import Dict, List, Tuple, Callable

from .bucket import GradientBucket
from ..core.config import BucketConfig

class BucketManager:
    """
    Manages the creation, grouping, and lifecycle of gradient buckets.

    It creates `GradientBucket` instances from a model's parameters and
    registers backward hooks to monitor when gradients are ready for reduction.
    """
    
    def __init__(self, config: BucketConfig):
        """
        Initializes the BucketManager.

        Args:
            config (BucketConfig): Configuration object specifying bucket
                parameters like capacity.
        """
        self.config = config
        # Stores GradientBucket instances, keyed by their ID.
        self.buckets: Dict[int, GradientBucket] = {}
        # Stores references to the registered hooks for later removal.
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def create_buckets(self, model: nn.Module) -> Dict[int, GradientBucket]:
        """
        Creates gradient buckets from the model's parameters.

        Parameters are grouped into buckets based on the configured maximum
        capacity (in MB). Parameters are iterated in reverse order (typically
        the order gradients are computed) to ensure efficient filling.

        Args:
            model (nn.Module): The model whose parameters will be bucketed.

        Returns:
            Dict[int, GradientBucket]: A dictionary of created buckets,
                indexed by bucket ID. Returns an empty dict if bucketing is
                disabled by the config.
        """
        # If gradient_as_bucket_view is False, bucketing is disabled.
        if not self.config.gradient_as_bucket_view:
            return {}
        
        capacity_bytes = self.config.capacity_mb * 1024 * 1024 # Convert MB to bytes
        buckets = {}
        current_params = [] # Parameters currently being added to the active bucket
        current_size = 0 # Current size of the active bucket in bytes
        bucket_id = 0
        
        # Iterate model parameters in reverse order. This matches the order
        # in which gradients are typically computed during the backward pass.
        for param in reversed(list(model.parameters())):
            # Only consider parameters that require gradients
            if not param.requires_grad:
                continue
            
            # Calculate the size of the parameter's gradient in bytes
            param_bytes = param.numel() * param.element_size()
            
            # If adding the current parameter exceeds bucket capacity, and the
            # current bucket is not empty, finalize the current bucket and start a new one.
            if current_params and (current_size + param_bytes > capacity_bytes):
                buckets[bucket_id] = GradientBucket(bucket_id, current_params[:])
                bucket_id += 1
                current_params = []
                current_size = 0
            
            current_params.append(param)
            current_size += param_bytes
        
        # After iterating all parameters, add any remaining parameters to a final bucket.
        if current_params:
            buckets[bucket_id] = GradientBucket(bucket_id, current_params[:])
        
        self.buckets = buckets
        return buckets
    
    def register_hooks(self, bucket_ready_callback: Callable[[int], None]) -> None:
        """
        Registers backward hooks on the gradients of parameters within each bucket.

        When a gradient for a parameter is computed, the hook will be triggered.
        When all gradients in a bucket are computed, or the backward pass
        reaches a certain point, the `bucket_ready_callback` is invoked.

        Args:
            bucket_ready_callback (Callable[[int], None]): A callback function
                that takes the `bucket_id` as an argument. This function is
                called when a bucket's gradients are ready for reduction.
        """
        # Clear any existing hooks before registering new ones
        self.remove_all_hooks()

        def make_hook(bucket_id: int):
            """
            Creates a closure for the backward hook. Each hook is associated
            with a specific bucket_id.
            """
            def hook(grad):
                # When any parameter in this bucket receives its gradient,
                # mark the bucket as having a gradient ready. Increment the
                # counter for gradients received in this bucket.
                bucket = self.buckets[bucket_id]
                bucket.increment_gradient_count()
                
                import torch.distributed as dist

                # If all gradients for this bucket have arrived, trigger the callback.
                if bucket.are_all_gradients_ready():
                    bucket_ready_callback(bucket_id)
                return grad
            return hook
        
        # Register the generated hooks for each parameter in each bucket
        for bucket_id, bucket in self.buckets.items():
            for param in bucket.parameters:
                if param.requires_grad:
                    # `register_hook` returns a handle that can be used to remove the hook
                    handle = param.register_hook(make_hook(bucket_id))
                    self._hooks.append(handle)
        
    def remove_all_hooks(self) -> None:
        """
        Removes all gradient hooks currently registered by this manager.

        This is important for proper cleanup, especially when the DDP wrapper
        is being destroyed or re-initialized with new buckets.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks = [] # Clear the list of handles
    
    def get_bucket_info(self) -> List[Tuple[int, int, float]]:
        """
        Retrieves summary information about all managed buckets.

        Returns:
            List[Tuple[int, int, float]]: A list of tuples, where each tuple
                contains (bucket_id, number_of_parameters_in_bucket, bucket_size_in_MB).
        """
        return [
            (bucket_id, len(bucket.parameters), bucket.size_mb())
            for bucket_id, bucket in self.buckets.items()
        ]
