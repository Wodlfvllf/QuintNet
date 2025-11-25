"""
Custom Distributed Data Parallel (DDP) Implementation

This module provides a modular and customizable implementation of Distributed
Data Parallel (DDP). It allows for flexible configuration of how gradients
are bucketed and reduced across distributed processes.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Distributed Data Parallel (DDP) is a common strategy for scaling model training
across multiple GPUs or nodes. Each process (typically one per GPU) holds a
replica of the model and processes a different batch of data. After the
forward and backward passes, the gradients from all model replicas are
averaged to ensure that all replicas maintain synchronized weights.

This `CustomDDP` implementation breaks down the DDP functionality into
pluggable components:

-   **`DistributedConfig`**: Defines basic distributed environment settings.
-   **`BucketConfig`**: Controls how model parameters are grouped into buckets
    for efficient gradient communication.
-   **`DistributedBackend`**: Abstracts the underlying communication library
    (e.g., PyTorch's `torch.distributed`).
-   **`BucketManager`**: Creates and manages the gradient buckets.
-   **`GradientReducer`**: Handles the all-reduce operation for gradients
    within each bucket.
-   **`ParameterBroadcaster`**: Ensures initial model parameters are identical
    across all replicas.

This modular design allows for easy experimentation with different
communication strategies and optimizations.

===============================================================================
"""

import torch.nn as nn
from typing import Optional, List, Tuple

from ..backends.base import DistributedBackend
from ..backends.torch_backend import TorchDistributedBackend
from ..components.bucket_manager import BucketManager
from ..components.gradient_reducer import GradientReducer
from ..components.parameter_broadcaster import ParameterBroadcaster
from .config import DistributedConfig, BucketConfig, ReductionStrategy

class CustomDDP(nn.Module):
    """
    A modular custom Distributed Data Parallel (DDP) implementation.

    This class wraps a `torch.nn.Module` and orchestrates the gradient
    synchronization across multiple processes using pluggable components
    for bucketing, reduction, and parameter broadcasting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        distributed_config: Optional[DistributedConfig] = None,
        bucket_config: Optional[BucketConfig] = None,
        backend: Optional[DistributedBackend] = None,
        reduction_strategy: ReductionStrategy = ReductionStrategy.MEAN,
    ):
        """
        Initializes the CustomDDP wrapper.

        Args:
            model (nn.Module): The model to be wrapped for data parallelism.
            distributed_config (Optional[DistributedConfig]): Configuration for
                the distributed environment. If None, a default is used.
            bucket_config (Optional[BucketConfig]): Configuration for gradient
                bucketing. If None, a default is used.
            backend (Optional[DistributedBackend]): The distributed communication
                backend to use. If None, `TorchDistributedBackend` is used.
            reduction_strategy (ReductionStrategy): The strategy to use for
                reducing gradients (e.g., MEAN for averaging).
        """
        super().__init__()
        
        self.model = model
        self.distributed_config = distributed_config or DistributedConfig()
        self.bucket_config = bucket_config or BucketConfig()
        self.backend = backend or TorchDistributedBackend() # Default to PyTorch's backend
        
        # Initialize core components for DDP functionality
        self.bucket_manager = BucketManager(self.bucket_config)
        self.gradient_reducer = GradientReducer(self.backend, self.distributed_config, reduction_strategy)
        self.parameter_broadcaster = ParameterBroadcaster(self.backend, self.distributed_config)
        
        # Perform initial setup steps
        self._setup()
    
    def _setup(self) -> None:
        """
        Performs the initial setup for the DDP wrapper.

        This includes creating gradient buckets, broadcasting initial parameters
        to ensure all model replicas are identical, and registering hooks
        to trigger gradient reduction.
        """
        # Create buckets for efficient gradient communication
        self.buckets = self.bucket_manager.create_buckets(self.model)
        
        # Ensure all model replicas start with the same parameters
        self.parameter_broadcaster.broadcast_parameters(self.model)
        
        # Register backward hooks to trigger gradient reduction when a bucket is full
        if self.buckets:
            self.bucket_manager.register_hooks(self._on_bucket_ready)
    
    def _on_bucket_ready(self, bucket_id: int) -> None:
        """
        Callback function invoked when a gradient bucket is ready for reduction.

        Args:
            bucket_id (int): The ID of the bucket that is ready.
        """
        bucket = self.buckets[bucket_id]
        bucket.is_ready = True # Mark the bucket as ready
        
        # Only reduce if the bucket actually contains gradients
        if bucket.has_gradients():
            self.gradient_reducer.reduce_bucket(bucket)
    
    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the wrapped model.

        All arguments are passed directly to the underlying model's forward method.
        """
        return self.model(*args, **kwargs)
    
    def remove_hooks(self) -> None:
        """
        Removes all gradient hooks registered by the `BucketManager`.

        This is important for proper cleanup, especially when the DDP wrapper
        is being destroyed or re-initialized.
        """
        self.bucket_manager.remove_all_hooks()
    
    def get_bucket_info(self) -> List[Tuple[int, int, float]]:
        """
        Retrieves information about the gradient buckets.

        Returns:
            List[Tuple[int, int, float]]: A list of tuples, where each tuple
                contains (bucket_id, number_of_parameters, total_memory_in_MB).
        """
        return self.bucket_manager.get_bucket_info()
    
    def __del__(self):
        """
        Destructor to ensure gradient hooks are removed when the CustomDDP
        object is garbage collected.
        """
        try:
            self.remove_hooks()
        except:
            # Ignore errors during cleanup, as the distributed environment
            # might already be torn down.
            pass
