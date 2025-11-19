"""
Distributed Communication Primitives

This module provides custom Autograd functions and helper utilities for
efficient inter-process communication in distributed training, particularly
for pipeline and tensor parallelism. It wraps `torch.distributed` operations
to integrate them seamlessly into PyTorch's autograd graph.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

In distributed training, communication between processes (GPUs) is a frequent
and performance-critical operation. PyTorch's `torch.distributed` provides
low-level primitives, but integrating them directly into a model's forward
and backward passes can be complex, especially when gradients need to be
communicated.

This module addresses this by:
-   **Custom Autograd Functions**: `Send`, `Recv`, `All_Gather`, `All_Reduce`,
    and `ReduceScatter` are implemented as `torch.autograd.Function` subclasses.
    This allows PyTorch's autograd engine to automatically handle the
    corresponding communication in the backward pass when these operations
    are used in the forward pass.
-   **Pipeline Communication Helpers**: `pipeline_communicate` and
    `bidirectional_pipeline_communicate` provide higher-level abstractions
    for point-to-point communication between pipeline stages, handling
    tensor shape/dtype metadata and non-blocking operations.

These primitives are essential for building efficient pipeline and tensor
parallelism implementations.

===============================================================================
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import os
from typing import Optional, Tuple, List, Any

# Global step counter for debugging verbose output
STEP = 0
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

class Send(torch.autograd.Function):
    """
    Custom Autograd Function for sending a tensor to a destination rank.

    In the forward pass, it sends the tensor. In the backward pass, it
    receives the gradient from the destination rank.
    """
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, dest_rank: int, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Sends a tensor to the specified destination rank.

        Args:
            ctx (Any): The context object to save information for backward pass.
            tensor (torch.Tensor): The tensor to send.
            dest_rank (int): The rank of the destination process.
            group (dist.ProcessGroup): The process group for communication.

        Returns:
            torch.Tensor: The original tensor (to maintain the computational graph).
        """
        # Store context for the backward pass
        ctx.dest_rank = dest_rank
        ctx.group = group
        ctx.tensor_shape = tensor.shape
        ctx.tensor_dtype = tensor.dtype

        # Ensure tensor is on the correct device and contiguous for sending
        tensor_to_send = tensor.contiguous()

        try:
            # 1. Send the number of dimensions (metadata for receiver)
            num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
            dist.send(tensor=num_dims, dst=dest_rank, group=group)

            # 2. Send the shape of the tensor (metadata for receiver)
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
            dist.send(tensor=shape_tensor, dst=dest_rank, group=group)

            # 3. Send the actual tensor data
            dist.send(tensor=tensor_to_send, dst=dest_rank, group=group)
            
        except Exception as e:
            print(f"Send.forward: ERROR on rank {dist.get_rank(group)} sending to {dest_rank} - {e}")
            raise
        
        # Return the original tensor to maintain the computational graph
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        Receives gradients from the destination rank during the backward pass.

        Args:
            ctx (Any): The context object with saved information.
            grad_output (torch.Tensor): The incoming gradient from the subsequent
                operation in the backward graph (not directly used here, as
                gradient is received from peer).

        Returns:
            Tuple[Optional[torch.Tensor], None, None]: The received gradient
                tensor for the input `tensor`, and None for other non-tensor inputs.
        """
        # Pre-allocate a tensor to receive the gradient with the expected shape and dtype
        grad_tensor = torch.zeros(
            ctx.tensor_shape,
            dtype=ctx.tensor_dtype,
            device=grad_output.device # Use the device of the incoming grad_output
        )
        
        try:
            # Receive the gradient tensor from the peer
            dist.recv(tensor=grad_tensor, src=ctx.dest_rank, group=ctx.group)
        except Exception as e:
            print(f"Send.backward: ERROR on rank {dist.get_rank(ctx.group)} receiving grad from {ctx.dest_rank} - {e}")
            raise
        
        # Return the received gradient for the input tensor.
        # None for dest_rank and group as they are not tensor inputs.
        return grad_tensor, None, None


class Recv(torch.autograd.Function):
    """
    Custom Autograd Function for receiving a tensor from a source rank.

    In the forward pass, it receives the tensor. In the backward pass, it
    sends the gradient back to the source rank.
    """
    @staticmethod
    def forward(ctx: Any, src_rank: int, device: torch.device, group: dist.ProcessGroup, dtype: torch.dtype) -> torch.Tensor:
        """
        Receives a tensor from the specified source rank.

        Args:
            ctx (Any): The context object to save information for backward pass.
            src_rank (int): The rank of the source process.
            device (torch.device): The device to place the received tensor on.
            group (dist.ProcessGroup): The process group for communication.
            dtype (torch.dtype): The expected data type of the received tensor.

        Returns:
            torch.Tensor: The received tensor.
        """
        # Store context for the backward pass
        ctx.src_rank = src_rank
        ctx.group = group
        ctx.dtype = dtype # Save dtype for backward pass gradient allocation

        try:
            # 1. Receive the number of dimensions (metadata)
            num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
            dist.recv(tensor=num_dims_tensor, src=src_rank, group=group)
            num_dims = num_dims_tensor.item()

            # 2. Receive the shape of the tensor (metadata)
            shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
            dist.recv(tensor=shape_tensor, src=src_rank, group=group)
            tensor_shape = shape_tensor.tolist()
            
            # 3. Allocate and receive the actual tensor data
            received_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
            dist.recv(tensor=received_tensor, src=src_rank, group=group)
            
        except Exception as e:
            print(f"Recv.forward: ERROR on rank {dist.get_rank(group)} receiving from {src_rank} - {e}")
            print(f"Recv.forward: Process group info - rank: {dist.get_rank(group)}, size: {dist.get_world_size(group)}")
            raise

        return received_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, None, None, None]:
        """
        Sends gradients back to the source rank during the backward pass.

        Args:
            ctx (Any): The context object with saved information.
            grad_output (torch.Tensor): The incoming gradient from the subsequent
                operation in the backward graph.

        Returns:
            Tuple[None, None, None, None]: None for all non-tensor inputs to forward.
        """
        # Send the gradient to the rank we received the tensor from
        dest_rank = ctx.src_rank
        
        try:
            # Ensure gradient is contiguous before sending
            grad_to_send = grad_output.contiguous()
            dist.send(tensor=grad_to_send, dst=dest_rank, group=ctx.group)
        except Exception as e:
            print(f"Recv.backward: ERROR on rank {dist.get_rank(ctx.group)} sending grad to {dest_rank} - {e}")
            raise
        
        # Return None for all inputs to the forward pass that are not tensors
        # (src_rank, device, group, dtype)
        return None, None, None, None


def pipeline_communicate(operation: str, 
                         pp_group: dist.ProcessGroup,
                         pp_rank: int,
                         device: torch.device, 
                         dtype: torch.dtype,
                         is_first_stage: bool = False,
                         is_last_stage: bool = False,
                         tensor: Optional[torch.Tensor] = None, 
                         shapes: Optional[Tuple[int, ...]] = None
                         ) -> Optional[torch.Tensor]:
    """
    Handles unidirectional point-to-point communication between pipeline stages.

    This function abstracts away the direct `Send` and `Recv` calls and
    applies boundary conditions for the first and last pipeline stages.

    Args:
        operation (str): The type of communication operation.
            One of ['recv_forward', 'send_forward', 'recv_backward', 'send_backward'].
        pp_group (dist.ProcessGroup): The process group for pipeline parallel communication.
        pp_rank (int): The rank of the current process within the pipeline parallel group.
        device (torch.device): The device to allocate tensors on.
        dtype (torch.dtype): The data type for tensors.
        is_first_stage (bool): True if the current rank is the first stage in the pipeline.
        is_last_stage (bool): True if the current rank is the last stage in the pipeline.
        tensor (Optional[torch.Tensor]): The tensor to send (required for send operations).
        shapes (Optional[Tuple[int, ...]]): The expected shape of the tensor to receive
            (required for receive operations).

    Returns:
        Optional[torch.Tensor]: The received tensor for 'recv' operations, None for 'send' operations.

    Raises:
        ValueError: If an unknown operation is specified.
    """
    global STEP
    global VERBOSE
    
    if operation == 'recv_forward':
        # First stage receives input from dataloader, not from previous stage
        if is_first_stage:
            return None
        # For receiving, we need to know the source rank
        src = pp_rank - 1
        # Use Recv autograd function to receive the tensor
        return Recv.apply(src, device, pp_group, dtype)
        
    elif operation == 'send_forward':
        # Last stage doesn't send forward activations (it computes loss)
        if is_last_stage:
            return None
        # For sending, we need to know the destination rank and the tensor to send
        dest = pp_rank + 1
        if tensor is None:
            raise ValueError("Tensor must be provided for send_forward operation.")
        # Use Send autograd function to send the tensor
        return Send.apply(tensor, dest, pp_group)
        
    elif operation == 'recv_backward':
        # Last stage doesn't receive backward gradients (it initiates backward pass)
        if is_last_stage:
            return None
        # For receiving, we need to know the source rank
        src = pp_rank + 1
        # Use Recv autograd function to receive the gradient
        return Recv.apply(src, device, pp_group, dtype)
        
    elif operation == 'send_backward':
        # First stage doesn't send backward gradients (it's the end of the backward pass)
        if is_first_stage:
            return None
        # For sending, we need to know the destination rank and the tensor to send
        dest = pp_rank - 1
        if tensor is None:
            raise ValueError("Tensor must be provided for send_backward operation.")
        # Use Send autograd function to send the gradient
        return Send.apply(tensor, dest, pp_group)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def bidirectional_pipeline_communicate(operation: str, 
                                       pp_group: dist.ProcessGroup, 
                                       pp_rank: int, 
                                       send_tensor: torch.Tensor, 
                                       recv_shapes: Tuple[int, ...], 
                                       device: torch.device, 
                                       dtype: torch.dtype,
                                       is_first_stage: bool = False,
                                       is_last_stage: bool = False
                                       ) -> Optional[torch.Tensor]:
    """
    Handles bidirectional communication between pipeline stages (send and recv simultaneously).

    This is typically used in the 1F1B schedule to overlap forward and backward
    communications, reducing pipeline bubbles.

    Args:
        operation (str): The type of communication operation.
            One of ['send_fwd_recv_bwd', 'send_bwd_recv_fwd'].
        pp_group (dist.ProcessGroup): The process group for pipeline parallel communication.
        pp_rank (int): The rank of the current process within the pipeline parallel group.
        send_tensor (torch.Tensor): The tensor to send.
        recv_shapes (Tuple[int, ...]): The expected shape of the tensor to receive.
        device (torch.device): The device to allocate the received tensor on.
        dtype (torch.dtype): The data type for tensors.
        is_first_stage (bool): True if the current rank is the first stage in the pipeline.
        is_last_stage (bool): True if the current rank is the last stage in the pipeline.

    Returns:
        Optional[torch.Tensor]: The received tensor. Returns None if the current
            stage is a boundary stage that doesn't participate in this bidirectional
            communication.
    """
    global STEP
    global VERBOSE
    
    is_fwd_send_bwd_recv = (operation == 'send_fwd_recv_bwd')
    
    # Boundary conditions: first/last stages don't do bidirectional communication
    # If sending forward, the last stage doesn't send.
    # If sending backward, the first stage doesn't send.
    if (is_fwd_send_bwd_recv and is_last_stage) or (not is_fwd_send_bwd_recv and is_first_stage):
        return None
    
    # Determine peer rank based on operation
    peer_rank = pp_rank + 1 if is_fwd_send_bwd_recv else pp_rank - 1
    
    # Allocate receive buffer
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    
    # Create bidirectional operations using P2POp
    # The order of operations in batch_isend_irecv matters for deadlock avoidance
    # but here we are doing send and recv to the same peer, so it's fine.
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor.contiguous(), peer_rank, group=pp_group),
        dist.P2POp(dist.irecv, recv_tensor, peer_rank, group=pp_group)
    ])
    
    if VERBOSE:
        send_dir = 'next' if is_fwd_send_bwd_recv else 'prev'
        recv_dir = 'prev' if is_fwd_send_bwd_recv else 'next' # Recv is always from the peer
        print(f"{operation} | rank {pp_rank} sending {send_dir} to {peer_rank} | "
              f"receiving {recv_dir} from {peer_rank} | "
              f"STEP={STEP} | RANK:{pp_rank}", flush=True)
    
    # Wait for both operations to complete
    for req in reqs:
        req.wait()
    torch.cuda.synchronize() # Ensure CUDA operations are complete
    
    if VERBOSE:
        STEP += 1
    
    return recv_tensor


class All_Gather(torch.autograd.Function):
    """
    Custom Autograd Function for `torch.distributed.all_gather`.

    Forward:
      - Takes a local tensor `x` (e.g., a sharded output from `ColumnParallelLinear`).
      - Performs `all_gather` to collect `x` from all ranks in the group.
      - Returns the concatenated tensor `[..., local_dim * world_size]`.

    Backward:
      - `mode='slice'`: Each rank slices `grad_output` to its local chunk (no communication).
        This is used when `grad_output` is replicated on all ranks.
      - `mode='reduce_scatter'`: Uses `reduce_scatter` across ranks. This is for
        advanced patterns where `grad_output` is not replicated on all ranks,
        or when a communication-based gradient aggregation is desired.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup, mode: str = "slice") -> torch.Tensor:
        """
        Performs the all-gather operation.

        Args:
            ctx (Any): The context object.
            x (torch.Tensor): The local tensor to be gathered.
            group (dist.ProcessGroup): The process group for all-gather.
            mode (str): The backward mode ('slice' or 'reduce_scatter').

        Returns:
            torch.Tensor: The concatenated tensor from all ranks.
        """
        # Save the group (non-tensor) and mode on ctx for backward.
        ctx.tp_group = group
        ctx.mode = mode

        # Save any tensor you need for backward (optional here, x is not modified)
        # ctx.save_for_backward(x) # Not strictly needed if x is not modified and its shape is derivable

        # Ensure tensor is contiguous and on the correct device
        out = x.contiguous()

        world_size = dist.get_world_size(group=group)
        gather_list = [torch.empty_like(out) for _ in range(world_size)]

        # all_gather: collect local tensors from all ranks into `gather_list`
        dist.all_gather(gather_list, out, group=group)

        # Concatenate the gathered tensors along the last dimension
        concat_output = torch.cat(gather_list, dim=-1).contiguous()
        return concat_output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        Handles the backward pass for the all-gather operation.

        Args:
            ctx (Any): The context object.
            grad_output (torch.Tensor): The incoming gradient for the concatenated output.

        Returns:
            Tuple[Optional[torch.Tensor], None, None]: The gradient for the
                local input `x`, and None for other non-tensor inputs.
        """
        # grad_output: [..., total_dim]
        tp_group = ctx.tp_group
        mode = ctx.mode

        world_size = dist.get_world_size(group=tp_group)
        rank = dist.get_rank(group=tp_group)

        # ----- Mode 1: slice (recommended when each rank has full grad_output) -----
        # This is typically used when the output of All_Gather is consumed by a
        # layer that is replicated across TP ranks (e.g., a non-parallel layer).
        if mode == "slice":
            # Split the incoming gradient into `world_size` chunks and return
            # the chunk corresponding to this rank. No communication is needed.
            grads = torch.chunk(grad_output, world_size, dim=-1)
            grad_local = grads[rank].contiguous()
            return grad_local, None, None  # gradient for (x, group, mode)

        # ----- Mode 2: reduce_scatter (use carefully) -----
        # This is for advanced patterns where `grad_output` is not replicated
        # on every rank, or when a communication-based gradient aggregation is desired.
        elif mode == "reduce_scatter":
            # Prepare a list of chunks (local) for reduce_scatter.
            # `grad_output` is split into `world_size` parts.
            grads = list(torch.chunk(grad_output, world_size, dim=-1))
            # Allocate output chunk for the reduce_scatter result.
            out_chunk = torch.empty_like(grads[0])
            # `reduce_scatter` will sum gradients element-wise across ranks and
            # scatter the results to the corresponding ranks.
            dist.reduce_scatter(out_chunk, grads, group=tp_group, op=dist.ReduceOp.SUM)
            # Note: If every rank had identical `grads`, this `out_chunk` will be
            # `sum(grads_i across ranks)`, i.e., multiplied by `world_size`.
            # Handle scaling outside if necessary.
            return out_chunk, None, None

        else:
            raise RuntimeError(f"Unknown All_Gather backward mode: {mode}")


class All_Reduce(torch.autograd.Function):
    """
    Custom Autograd Function for `torch.distributed.all_reduce`.

    Forward:
      - Takes a local tensor `x`.
      - Performs `all_reduce` (sum) across all ranks in the group.
      - Returns the reduced tensor.

    Backward:
      - Since `all_reduce` sums contributions, the gradient for each input
        is simply the incoming `grad_output` (which is already the sum of
        gradients from all downstream operations). No further communication
        is needed in the backward pass for the input `x`.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Performs the all-reduce operation (sum).

        Args:
            ctx (Any): The context object.
            x (torch.Tensor): The local tensor to be all-reduced.
            group (dist.ProcessGroup): The process group for all-reduce.

        Returns:
            torch.Tensor: The all-reduced tensor.
        """
        # Save the group (non-tensor) on ctx for backward.
        ctx.tp_group = group

        # Ensure tensor is contiguous and on the correct device
        out = x.contiguous()

        # all-reduce: sum across TP group
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)

        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Handles the backward pass for the all-reduce operation.

        Args:
            ctx (Any): The context object.
            grad_output (torch.Tensor): The incoming gradient for the all-reduced output.

        Returns:
            Tuple[torch.Tensor, None]: The gradient for the input `x`, and None
                for the non-tensor `group` input.
        """
        # For an all-reduce (sum) in the forward, the gradient for each input
        # is simply the incoming grad_output. No communication is needed here.
        return grad_output, None


class ReduceScatter(torch.autograd.Function):
    """
    Custom Autograd Function for `torch.distributed.reduce_scatter`.

    Forward:
      - Takes a local tensor `x`.
      - Splits `x` into `world_size` chunks.
      - Performs `reduce_scatter` (sum) across ranks, where each rank receives
        the sum of the corresponding chunk from all other ranks.
      - Returns the local chunk of the reduced and scattered tensor.

    Backward:
      - Performs `all_gather` on the incoming `grad_output` to reconstruct
        the full gradient for the original input `x`.
    """
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Performs the reduce-scatter operation.

        Args:
            ctx (Any): The context object.
            x (torch.Tensor): The local tensor to be reduced and scattered.
            group (dist.ProcessGroup): The process group for reduce-scatter.

        Returns:
            torch.Tensor: The local chunk of the reduced and scattered tensor.
        """
        ctx.tp_group = group
        world_size = dist.get_world_size(group)
        
        # Split input tensor into `world_size` chunks
        input_list = list(torch.chunk(x, world_size, dim=-1))
        # Allocate output tensor for the local chunk
        output = torch.empty_like(input_list[0])
        
        # Perform reduce-scatter (sum)
        dist.reduce_scatter(output, input_list, group=group, op=dist.ReduceOp.SUM)
        return output
    
    @staticmethod 
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Handles the backward pass for the reduce-scatter operation.

        Args:
            ctx (Any): The context object.
            grad_output (torch.Tensor): The incoming gradient for the local
                chunk of the reduced and scattered output.

        Returns:
            Tuple[torch.Tensor, None]: The gradient for the original input `x`,
                and None for the non-tensor `group` input.
        """
        # To get the gradient for the original input `x`, we need to all-gather
        # the `grad_output` from all ranks.
        world_size = dist.get_world_size(ctx.tp_group)
        gather_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gather_list, grad_output, group=ctx.tp_group)
        
        # Concatenate the gathered gradients to form the full gradient for `x`.
        return torch.cat(gather_list, dim=-1), None