"""
Gradient Bucket Implementation

This module defines the `GradientBucket` class, which represents a single
collection of model parameters whose gradients will be aggregated and
all-reduced together in a Distributed Data Parallel (DDP) setup.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

A `GradientBucket` is a fundamental component of gradient bucketing. It
encapsulates a list of `nn.Parameter` objects and provides methods to:

-   Calculate the total memory size of the gradients it contains.
-   Check if all parameters within the bucket have received their gradients.
-   Flatten all gradients into a single contiguous tensor for efficient
    communication.
-   Unflatten a received tensor back into the individual parameter gradients.
-   Manage backward hooks that signal when gradients are ready.

This class works in conjunction with `BucketManager` to orchestrate the
gradient communication process.

===============================================================================
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Callable

class GradientBucket:
    """
    Manages a single gradient bucket, grouping parameters for efficient
    gradient communication in DDP.
    """
    
    def __init__(self, bucket_id: int, parameters: List[nn.Parameter]):
        """
        Initializes a GradientBucket.

        Args:
            bucket_id (int): A unique identifier for this bucket.
            parameters (List[nn.Parameter]): A list of `nn.Parameter` objects
                whose gradients belong to this bucket.
        """
        self.bucket_id = bucket_id
        self.parameters = parameters
        self.is_ready = False # Flag to indicate if the bucket is ready for reduction
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = [] # Stores handles for registered hooks
        self._gradients_received_count = 0 # Counter for gradients received in this bucket
    
    def size_bytes(self) -> int:
        """
        Calculates the total memory size (in bytes) of all gradients
        associated with the parameters in this bucket.

        Returns:
            int: The total size in bytes.
        """
        # Sum the number of elements * element size for each parameter's gradient
        return sum(p.numel() * p.element_size() for p in self.parameters if p.requires_grad)
    
    def size_mb(self) -> float:
        """
        Calculates the total memory size (in megabytes) of all gradients
        associated with the parameters in this bucket.

        Returns:
            float: The total size in megabytes.
        """
        return self.size_bytes() / (1024 ** 2)
    
    def increment_gradient_count(self) -> None:
        """
        Increments the counter for gradients received in this bucket.
        This is typically called by a backward hook.
        """
        self._gradients_received_count += 1

    def are_all_gradients_ready(self) -> bool:
        """
        Checks if all parameters in this bucket that require gradients
        have received their gradients.

        Returns:
            bool: True if all gradients are ready, False otherwise.
        """
        # Compare the count of received gradients with the total number of
        # parameters in the bucket that require gradients.
        num_params_with_grad = sum(1 for p in self.parameters if p.requires_grad)
        return self._gradients_received_count == num_params_with_grad
    
    def has_gradients(self) -> bool:
        """
        Checks if at least one parameter in the bucket has a gradient.

        Returns:
            bool: True if any parameter has a gradient, False otherwise.
        """
        return any(p.grad is not None for p in self.parameters if p.requires_grad)
    
    def flatten_gradients(self) -> Tuple[torch.Tensor, torch.device]:
        """
        Flattens all gradients of parameters in the bucket into a single
        contiguous 1D tensor.

        Returns:
            Tuple[torch.Tensor, torch.device]: A tuple containing the flattened
                gradient tensor and the device it resides on.

        Raises:
            ValueError: If no gradients are found in the bucket.
        """
        flat_grads = []
        devices = []
        
        for p in self.parameters:
            if p.grad is not None:
                # Detach and flatten the gradient
                flat = p.grad.detach().view(-1)
                flat_grads.append(flat)
                devices.append(p.grad.device)
        
        if len(flat_grads) == 0:
            raise ValueError(f"No gradients found in bucket {self.bucket_id}")
        
        # Ensure all gradients are on the same device before concatenating
        device = devices[0]
        flat_grads = [g.to(device) for g in flat_grads]
        return torch.cat(flat_grads), device
    
    def unflatten_gradients(self, flattened_tensor: torch.Tensor) -> None:
        """
        Unflattens a 1D tensor back into the individual gradients of the
        parameters in this bucket.

        Args:
            flattened_tensor (torch.Tensor): The 1D tensor containing the
                flattened gradients.
        """
        offset = 0
        for p in self.parameters:
            if p.grad is None:
                continue # Skip parameters that don't have gradients
            
            numel = p.grad.numel()
            # Extract the chunk corresponding to this parameter's gradient
            chunk = flattened_tensor[offset:offset + numel]
            # Copy the data from the chunk into the parameter's gradient,
            # reshaping it back to the original gradient's shape.
            p.grad.data.copy_(chunk.view_as(p.grad).to(p.grad.device))
            offset += numel
    
    def register_hooks(self, hook_fn: Callable) -> None:
        """
        Registers a backward hook function for each parameter in the bucket.

        Args:
            hook_fn (Callable): The function to be called when a gradient
                for a parameter in this bucket is computed.
        """
        for p in self.parameters:
            if p.requires_grad:
                # Register the hook and store its handle for later removal
                handle = p.register_hook(hook_fn)
                self._hook_handles.append(handle)
    
    def remove_hooks(self) -> None:
        """
        Removes all backward hooks previously registered for parameters
        in this bucket.
        """
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                # Ignore errors if hook was already removed or invalid
                pass
        self._hook_handles = [] # Clear the list of handles
