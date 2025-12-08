"""
Model Rewriting Utilities for Tensor Parallelism

This module provides functions to automatically convert standard PyTorch models
to tensor parallel versions by replacing linear layers with their parallel
equivalents (`ColumnParallelLinear`, `RowParallelLinear`).

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The core idea of tensor parallelism is to shard the weights of individual
layers across multiple devices. This module implements a "model rewriting"
approach, where a given model's `nn.Linear` layers are identified and
replaced with custom `ColumnParallelLinear` or `RowParallelLinear` layers.

The `apply_tensor_parallel` function recursively traverses the model's
module hierarchy. When it encounters an `nn.Linear` layer, it determines
how to shard its weights (either by columns for `out_features` or by rows
for `in_features`) and replaces the original layer with the appropriate
tensor-parallel counterpart.

This process happens *in-place* on the model, effectively transforming
a standard model into a tensor-parallel one.

===============================================================================
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .layers import ColumnParallelLinear, RowParallelLinear
from typing import Optional


def apply_tensor_parallel(model: nn.Module, 
                          tp_size: int,
                          tp_rank: int, 
                          tp_group: dist.ProcessGroup,
                          device : torch.device,
                          gather_output: bool = True, 
                          sync_gradients: bool = True, 
                          method_of_parallelism: str = "column") -> nn.Module:
    """
    Applies tensor parallelism to a given model by replacing `nn.Linear` layers
    with `ColumnParallelLinear` or `RowParallelLinear`.

    This function recursively traverses the model's modules and replaces
    standard linear layers with their tensor-parallel equivalents, sharding
    their weights across the tensor parallel group.

    Args:
        model (nn.Module): The model to parallelize. This model will be modified in-place.
        tp_size (int): The total number of ranks in the tensor parallel group.
        tp_rank (int): The rank of the current process within the tensor parallel group.
        tp_group (dist.ProcessGroup): The communication group for tensor parallelism.
        device (torch.device): The device (e.g., 'cuda:0') where the sharded
            layers for this rank should reside.
        gather_output (bool): For `ColumnParallelLinear`, whether to gather
            the output from all ranks. Defaults to True.
        sync_gradients (bool): For `ColumnParallelLinear`, whether to synchronize
            gradients during the backward pass. Defaults to True.
        method_of_parallelism (str): Specifies how to shard the linear layers.
            Can be "column" (shards `out_features`) or "row" (shards `in_features`).
            Defaults to "column".

    Returns:
        nn.Module: The modified model with tensor-parallel linear layers.
    """
    tp_group = tp_group
    tp_rank = tp_rank
    tp_world_size = tp_size
    
    local_device = device

    def replace_linear(module: nn.Module, module_path: str = ""):
        """
        Recursively finds and replaces `nn.Linear` modules within the model.
        """
        for name, child in list(module.named_children()):
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child, nn.Linear):
                in_f, out_f = child.in_features, child.out_features

                if method_of_parallelism == "column":
                    # Shard the output features (columns of the weight matrix)
                    if out_f % tp_world_size != 0:
                        continue  # Skip this layer if not divisible

                    cols_per_rank = out_f // tp_world_size

                    start = tp_rank * cols_per_rank
                    end   = (tp_rank + 1) * cols_per_rank

                    # Slice the weight and bias tensors for this rank
                    weight_slice = child.weight[start:end, :]  # (cols_per_rank, in_f)
                    bias_slice = None
                    if child.bias is not None:
                        bias_slice = child.bias[start:end]
                    
                    # Create the ColumnParallelLinear replacement
                    shard = ColumnParallelLinear(
                        local_device=local_device,
                        tp_group=tp_group,
                        in_features=in_f,
                        out_features_per_rank=cols_per_rank,
                        weight_slice=weight_slice,
                        bias_slice=bias_slice,
                        gather_output=gather_output,
                        sync_gradients=sync_gradients,
                    )
                elif method_of_parallelism == "row":
                    # Shard the input features (rows of the weight matrix)
                    if in_f % tp_world_size != 0:
                        continue  # Skip this layer if not divisible
                        
                    rows_per_rank = in_f // tp_world_size
                    start = tp_rank * rows_per_rank
                    end   = (tp_rank + 1) * rows_per_rank
                    
                    # Slice the weight tensor for this rank
                    weight_slice = child.weight[:, start:end]  # (out_f, rows_per_rank)
                    
                    # Bias handling for RowParallelLinear: only the first rank
                    # in the TP group stores and adds the bias to avoid duplication
                    # and ensure correct all-reduce summation.
                    bias_slice = None
                    if child.bias is not None:
                        if tp_rank == 0:
                            bias_slice = child.bias
                        else:
                            bias_slice = None # Other ranks do not have bias
                        
                    # Create the RowParallelLinear replacement
                    shard = RowParallelLinear(
                        local_device=local_device,
                        tp_group=tp_group,
                        in_features_per_rank=rows_per_rank,
                        out_features=out_f,
                        weight_slice=weight_slice,
                        bias_slice=bias_slice,
                        input_is_parallel=False,  # Assuming input is not sharded initially
                    )
                else:
                    raise ValueError(f"Unknown method_of_parallelism: {method_of_parallelism}. Must be 'column' or 'row'.")

                # Replace the original nn.Linear module with the tensor-parallel shard
                setattr(module, name, shard)

            else:
                # Recursively apply to child modules
                replace_linear(child, current_path)

    # Start the recursive replacement process from the root of the model
    replace_linear(model)
    
    # Ensure all ranks have finished replacing layers before proceeding
    dist.barrier()
    
    # Print a confirmation message on rank 0
    if dist.get_rank() == 0:
        pass
    
    return model
