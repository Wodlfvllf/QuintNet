import torch
import torch.nn as nn
import torch.distributed as dist
from .processgroup import ProcessGroupManager

class All_Gather(torch.autograd.Function):
    """
    Forward:
      - local x: shape [..., local_dim]
      - returns concatenated tensor [..., local_dim * world_size]

    Backward (mode='slice'): each rank slices grad_output to its local chunk (no comm).
    Backward (mode='reduce_scatter'): uses reduce_scatter across ranks. Use only if
      you know grad_output is not replicated on all ranks (or you want comm-based behavior).
    """

    @staticmethod
    def forward(ctx, x, group, mode="slice"):
        # Save the group (non-tensor) and mode on ctx for backward.
        ctx.tp_group = group
        ctx.mode = mode

        # Save any tensor you need for backward (optional here)
        ctx.save_for_backward(x)

        # ensure contiguous and device-correct
        out = x.contiguous()

        world_size = dist.get_world_size(group=group)
        gather_list = [torch.empty_like(out) for _ in range(world_size)]

        # all_gather: collect local tensors from all ranks
        dist.all_gather(gather_list, out, group=group)

        concat_output = torch.cat(gather_list, dim=-1).contiguous()
        return concat_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [..., total_dim]
        tp_group = ctx.tp_group
        mode = ctx.mode

        world_size = dist.get_world_size(group=tp_group)
        rank = dist.get_rank(group=tp_group)

        # ----- Mode 1: slice (recommended when each rank has full grad_output) -----
        if mode == "slice":
            # split into world_size chunks and return this rank's chunk
            grads = torch.chunk(grad_output, world_size, dim=-1)
            grad_local = grads[rank].contiguous()
            return grad_local, None, None  # gradient for (x, group, mode)

        # ----- Mode 2: reduce_scatter (use carefully) -----
        # This is for advanced patterns where full grad_output is not replicated on every rank
        elif mode == "reduce_scatter":
            # prepare list of chunks (local) for reduce_scatter
            grads = list(torch.chunk(grad_output, world_size, dim=-1))
            # allocate output chunk
            out_chunk = torch.empty_like(grads[0])
            # reduce_scatter will reduce (op=SUM) elementwise across ranks and scatter outputs
            dist.reduce_scatter(out_chunk, grads, group=tp_group, op=dist.ReduceOp.SUM)
            # Note: if every rank had identical grads, this out_chunk will be sum(grads_i across ranks)
            # i.e., multiplied by world_size. Handle scaling outside if necessary.
            return out_chunk, None, None

        else:
            raise RuntimeError(f"Unknown All_Gather backward mode: {mode}")

# ColumnParallelLinear using All_Gather.apply
class ColumnParallelLinear(nn.Module):
    """
    Holds a column shard of nn.Linear (split across out_features).
    If gather_output=True, returns the concatenated full output (each rank holds full).
    If gather_output=False, returns local output (for following row-parallel layers).
    """

    def __init__(self, local_device, tp_group, in_features, out_features_per_rank,
                 weight_slice, bias_slice=None, gather_output=True, sync_gradients=True,
                 gather_mode="slice"):
        super().__init__()
        self.device = local_device
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.sync_gradients = sync_gradients
        self.gather_mode = gather_mode  # "slice" or "reduce_scatter"

        # Create a local linear with out_features_per_rank
        self.proj = nn.Linear(in_features, out_features_per_rank, bias=(bias_slice is not None),
                              device=self.device)

        # Copy weights and bias into the parameter (no-grad)
        with torch.no_grad():
            self.proj.weight.copy_(weight_slice.to(self.device))
            if bias_slice is not None:
                self.proj.bias.copy_(bias_slice.to(self.device))

        # Ensure gradients are tracked
        self.proj.weight.requires_grad_(True)
        if self.proj.bias is not None:
            self.proj.bias.requires_grad_(True)

        # Prefer autograd function for comms (no hook), so we do not register backward hooks here.

    def forward(self, x):
        # Ensure input on proper device
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        local_out = self.proj(x)  # shape (B, out_features_per_rank)

        if self.gather_output:
            # IMPORTANT: use .apply to create autograd node
            return All_Gather.apply(local_out, self.tp_group, self.gather_mode)
        else:
            # Useful if next layer expects local shard only
            return local_out

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