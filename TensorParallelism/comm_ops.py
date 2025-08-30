"""
Communication operations for tensor parallelism.

This module provides PyTorch autograd functions for distributed communication
operations including All_Gather, All_Reduce, and ReduceScatter.
"""

import torch
import torch.distributed as dist


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


class All_Reduce(torch.autograd.Function):
    """
    Forward:
      - local x: shape [..., dim]
      - returns reduced tensor [..., dim] (SUM across TP group)

    Backward:
      - each rank gets identical grad_output, so just return it (no comm)
    """

    @staticmethod
    def forward(ctx, x, group):
        # Save the group (non-tensor) on ctx for backward.
        ctx.tp_group = group

        # ensure contiguous and device-correct
        out = x.contiguous()

        # all-reduce: sum across TP group
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # gradient for (x, group)


class ReduceScatter(torch.autograd.Function):
    """Reduce-scatter operation for more efficient gradient communication"""
    
    @staticmethod
    def forward(ctx, x, group):
        ctx.tp_group = group
        world_size = dist.get_world_size(group)
        
        # Split input into chunks
        input_list = list(torch.chunk(x, world_size, dim=-1))
        output = torch.empty_like(input_list[0])
        
        # Reduce-scatter
        dist.reduce_scatter(output, input_list, group=group, op=dist.ReduceOp.SUM)
        return output
    
    @staticmethod 
    def backward(ctx, grad_output):
        # All-gather gradients
        world_size = dist.get_world_size(ctx.tp_group)
        gather_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gather_list, grad_output, group=ctx.tp_group)
        return torch.cat(gather_list, dim=-1), None