"""
Model rewriting utilities for tensor parallelism.

This module provides functions to automatically convert standard PyTorch models
to tensor parallel versions by replacing linear layers with their parallel equivalents.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from .processgroup import ProcessGroupManager
from .layers import ColumnParallelLinear, RowParallelLinear


def apply_tensor_parallel(model: nn.Module, tp_size: int, gather_output=True, sync_gradients=True, method_of_parallelism="column"):
    """
    Replace every nn.Linear with a ColumnParallelLinear that holds its out_feature shard.
    Assumes dist.init_process_group() is already called and we're doing pure TP (no DP).
    
    Args:
        model: The model to parallelize
        tp_size: Tensor parallel size
        gather_output: Whether to gather outputs (True for most cases)
        sync_gradients: Whether to synchronize gradients (True when all ranks see same data)
        method_of_parallelism: "column" or "row" parallelism method
    """
    pgm = ProcessGroupManager(tp_size)
    tp_group = pgm.get_group()
    tp_rank = pgm.get_tp_rank()
    tp_world_size = pgm.get_tp_world_size()
    
    # Get the current device instead of global rank
    current_device = torch.cuda.current_device()
    local_device = torch.device(f"cuda:{current_device}")

    def replace_linear(module: nn.Module, module_path=""):
        for name, child in list(module.named_children()):
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child, nn.Linear):
                in_f, out_f = child.in_features, child.out_features

                if method_of_parallelism == "column":
                    if out_f % tp_world_size != 0:
                        print(f"Warning: {current_path} out_features {out_f} not divisible by tp_size {tp_world_size}")
                        continue  # Skip this layer

                    cols_per_rank = out_f // tp_world_size

                    start = tp_rank * cols_per_rank
                    end   = (tp_rank + 1) * cols_per_rank

                    weight_slice = child.weight[start:end, :]  # (cols_per_rank, in_f)
                    bias_slice = None
                    if child.bias is not None:
                        bias_slice = child.bias[start:end]
                    
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
                else:
                    if in_f % tp_world_size != 0:
                        print(f"Warning: {current_path} in_features {in_f} not divisible by tp_size {tp_world_size}")
                        continue  # Skip this layer
                        
                    rows_per_rank = in_f // tp_world_size
                    start = tp_rank * rows_per_rank
                    end   = (tp_rank + 1) * rows_per_rank
                    weight_slice = child.weight[:, start:end]  # (out_f, rows_per_rank)
                    bias_slice = None
                    if child.bias is not None:
                        bias_slice = child.bias
                        
                    shard = RowParallelLinear(
                        local_device=local_device,
                        tp_group=tp_group,
                        in_features_per_rank=rows_per_rank,
                        out_features=out_f,
                        weight_slice=weight_slice,
                        bias_slice=bias_slice,
                        input_is_parallel=False,  # assuming input is not sharded
                    )

                # Swap in-place
                setattr(module, name, shard)
                # print(f"Replaced {current_path}: {in_f} -> {out_f} (rank {tp_rank} gets cols {start}:{end})")

            else:
                replace_linear(child, current_path)

    replace_linear(model)
    
    # Ensure all ranks have finished replacing layers
    dist.barrier()
    
    if dist.get_rank() == 0:
        print(f"Applied tensor parallelism with tp_size={tp_size}")
        print(f"Gradient sync: {'enabled' if sync_gradients else 'disabled'}")
    
    return model