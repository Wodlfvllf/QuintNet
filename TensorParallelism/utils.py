import torch
import torch.nn as nn
import torch.distributed as dist
from .processgroup import ProcessGroupManager

def all_gather_cat_lastdim(local_tensor: torch.Tensor, group):
    """All-gather local shard across `group` and concat on last dim."""
    world_size = dist.get_world_size(group=group)
    # Preallocate buffers on the same device and with the same shape as local shard
    gather_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, local_tensor, group=group)
    return torch.cat(gather_list, dim=-1)  # concat feature shards


class ColumnParallelLinear(nn.Module):
    """
    Holds a column shard of an nn.Linear (split along out_features).
    Returns either the local shard or the concatenated full output depending on `gather_output`.
    """
    def __init__(self, local_device, tp_group, in_features, out_features_per_rank,
                 weight_slice, bias_slice, gather_output=True, sync_gradients=True):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.sync_gradients = sync_gradients  # New parameter

        self.proj = nn.Linear(in_features, out_features_per_rank, bias=(bias_slice is not None),
                              device=self.device)

        # Copy weights and ensure requires_grad=True
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
            if bias_slice is not None:
                self.proj.bias.copy_(bias_slice.to(self.device))
        
        # CRITICAL: Ensure gradients are enabled
        self.proj.weight.requires_grad_(True)
        if self.proj.bias is not None:
            self.proj.bias.requires_grad_(True)
        
        # Register backward hook for gradient synchronization
        if self.sync_gradients:
            self.proj.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, module, grad_input, grad_output):
        """Synchronize gradients across TP group since all ranks see same data"""
        if module.weight.grad is not None:
            dist.all_reduce(module.weight.grad, op=dist.ReduceOp.SUM, group=self.tp_group)
            module.weight.grad.div_(dist.get_world_size(self.tp_group))
        
        if module.bias is not None and module.bias.grad is not None:
            dist.all_reduce(module.bias.grad, op=dist.ReduceOp.SUM, group=self.tp_group)
            module.bias.grad.div_(dist.get_world_size(self.tp_group))

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        local_out = self.proj(x)            # (B, out_features_per_rank)
        if self.gather_output:
            return all_gather_cat_lastdim(local_out, self.tp_group)  # (B, total_out_features)
        else:
            return local_out  # Useful if next layer is row-parallel


def apply_tensor_parallel(model: nn.Module, tp_size: int, gather_output=True, sync_gradients=True):
    """
    Replace every nn.Linear with a ColumnParallelLinear that holds its out_feature shard.
    Assumes dist.init_process_group() is already called and we're doing pure TP (no DP).
    
    Args:
        model: The model to parallelize
        tp_size: Tensor parallel size
        gather_output: Whether to gather outputs (True for most cases)
        sync_gradients: Whether to synchronize gradients (True when all ranks see same data)
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

                # Swap in-place
                setattr(module, name, shard)
                print(f"Replaced {current_path}: {in_f} -> {out_f} (rank {tp_rank} gets cols {start}:{end})")

            else:
                replace_linear(child, current_path)

    replace_linear(model)
    
    # Ensure all ranks have finished replacing layers
    dist.barrier()
    
    if dist.get_rank() == 0:
        print(f"Applied tensor parallelism with tp_size={tp_size}")
        print(f"Gradient sync: {'enabled' if sync_gradients else 'disabled'}")
    
    return model


# Optional: Add row-parallel linear for future use
class RowParallelLinear(nn.Module):
    """
    Holds a row shard of an nn.Linear (split along in_features).
    Expects input to be already sharded and reduces output across TP group.
    """
    def __init__(self, local_device, tp_group, in_features_per_rank, out_features,
                 weight_slice, bias_slice=None, input_is_parallel=True):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel

        self.proj = nn.Linear(in_features_per_rank, out_features, bias=False, device=self.device)
        
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
        
        # Bias only on first rank to avoid duplication
        if bias_slice is not None and dist.get_rank(self.tp_group) == 0:
            self.bias = nn.Parameter(bias_slice.to(self.device))
        else:
            self.bias = None

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        local_out = self.proj(x)
        
        # All-reduce across TP group
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=self.tp_group)
        
        if self.bias is not None:
            local_out = local_out + self.bias
            
        return local_out