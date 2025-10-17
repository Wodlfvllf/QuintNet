import torch
import torch.distributed as dist
import torch.nn as nn
import os

class Send(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dest_rank, group):
        """
        Send a tensor to the destination rank.
        """
        # print(f"Send.forward: rank {dist.get_rank(group)} sending to rank {dest_rank}")
        # print(f"Send.forward: tensor shape {tensor.shape}, device {tensor.device}, dtype {tensor.dtype}")
        
        # Ensure tensor is on the correct device and contiguous
        tensor_to_send = tensor.contiguous()

        # Store context for the backward pass
        ctx.dest_rank = dest_rank
        ctx.group = group
        ctx.tensor_shape = tensor.shape
        ctx.tensor_dtype = tensor.dtype

        try:
            # 1. Send the number of dimensions
            num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
            # print(f"Send.forward: sending num_dims {num_dims.item()}")
            dist.send(tensor=num_dims, dst=dest_rank, group=group)

            # 2. Send the shape
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
            # print(f"Send.forward: sending shape {shape_tensor.tolist()}")
            dist.send(tensor=shape_tensor, dst=dest_rank, group=group)

            # 3. Send the tensor data
            # print(f"Send.forward: sending tensor data...")
            dist.send(tensor=tensor_to_send, dst=dest_rank, group=group)
            # print(f"Send.forward: tensor data sent successfully")
            
        except Exception as e:
            print(f"Send.forward: ERROR - {e}")
            raise
        
        # Return the original tensor to maintain the computational graph
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Receive gradients from the destination rank during the backward pass.
        """
        # print(f"Send.backward: rank {dist.get_rank(ctx.group)} waiting for grad from rank {ctx.dest_rank}")
        
        # Pre-allocate a tensor to receive the gradient
        grad_tensor = torch.zeros(
            ctx.tensor_shape,
            dtype=ctx.tensor_dtype,
            device=grad_output.device
        )
        
        try:
            # print(f"Send.backward: receiving gradient tensor...")
            dist.recv(tensor=grad_tensor, src=ctx.dest_rank, group=ctx.group)
            # print(f"Send.backward: gradient received successfully, norm: {grad_tensor.norm().item()}")
        except Exception as e:
            print(f"Send.backward: ERROR - {e}")
            raise
        
        return grad_tensor, None, None


class Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_rank, device, group, dtype):
        """
        Receive a tensor from the source rank.
        """
        # print(f"Recv.forward: rank {dist.get_rank(group)} receiving from rank {src_rank}")
        
        # Store context for the backward pass
        ctx.src_rank = src_rank
        ctx.group = group
        ctx.dtype = dtype

        try:
            # 1. Receive the number of dimensions
            num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
            # print(f"Recv.forward: waiting for num_dims...")
            dist.recv(tensor=num_dims_tensor, src=src_rank, group=group)
            num_dims = num_dims_tensor.item()
            # print(f"Recv.forward: received num_dims: {num_dims}")

            # 2. Receive the shape
            shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
            # print(f"Recv.forward: waiting for shape...")
            dist.recv(tensor=shape_tensor, src=src_rank, group=group)
            tensor_shape = shape_tensor.tolist()
            # print(f"Recv.forward: received shape: {tensor_shape}")
            
            # 3. Receive the tensor data
            # print(f"Recv.forward: waiting for tensor data...")
            received_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
            dist.recv(tensor=received_tensor, src=src_rank, group=group)
            # print(f"Recv.forward: received tensor successfully, norm: {received_tensor.norm().item()}")
            
        except Exception as e:
            print(f"Recv.forward: ERROR - {e}")
            print(f"Recv.forward: Process group info - rank: {dist.get_rank(group)}, size: {dist.get_world_size(group)}")
            raise

        return received_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Send gradients back to the source rank during the backward pass.
        """
        # print(f"Recv.backward: rank {dist.get_rank(ctx.group)} sending grad to rank {ctx.src_rank}")
        # print(f"Recv.backward: grad_output norm: {grad_output.norm().item()}")
        
        # Send the gradient to the rank we received the tensor from
        dest_rank = ctx.src_rank
        
        try:
            # Ensure gradient is contiguous before sending
            grad_to_send = grad_output.contiguous()
            dist.send(tensor=grad_to_send, dst=dest_rank, group=ctx.group)
            # print(f"Recv.backward: gradient sent successfully")
        except Exception as e:
            print(f"Recv.backward: ERROR - {e}")
            raise
        
        return None, None, None
    

"""
Communication operations for pipeline parallelism.
Handles point-to-point communication between pipeline stages.
"""

import os
import torch
import torch.distributed as dist


# Global step counter for debugging
STEP = 0
VERBOSE = os.environ.get("VERBOSE", "0") == "1"


def pipeline_communicate(operation, 
                         pp_group,
                         pp_rank,
                         device, 
                         dtype,
                         is_first_stage=False,
                         is_last_stage=False,
                         tensor=None, 
                         shapes=None
                         ):
    """
    Handles unidirectional communication between pipeline stages.
    
    Args:
        operation: One of ['recv_forward', 'send_forward', 'recv_backward', 'send_backward']
        pp_group: Pipeline parallel process group
        pp_rank: Rank within the pipeline parallel group
        device: Device to allocate tensors on
        dtype: Data type for tensors
        tensor: Tensor to send (for send operations)
        shapes: Shape tuple for receiving tensors (for recv operations)
    
    Returns:
        Received tensor for recv operations, None for send operations
    """
    global STEP
    global VERBOSE
    
    if operation == 'recv_forward':
        # First stage receives input from dataloader, not from previous stage
        if is_first_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pp_rank-1
        
    elif operation == 'send_forward':
        # Last stage doesn't send forward activations (outputs loss instead)
        if is_last_stage:
            return
        dest = pp_rank+1
        
    elif operation == 'recv_backward':
        # Last stage doesn't receive backward gradients (computes loss gradient)
        if is_last_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pp_rank+1
        
    elif operation == 'send_backward':
        # First stage doesn't send backward gradients
        if is_first_stage:
            return
        dest = pp_rank-1
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    
    # Create point-to-point operation
    op = dist.P2POp(
        dist.isend if is_send else dist.irecv,
        tensor,
        peer_rank,
        group=pp_group
    )
    
    if VERBOSE:
        direction = '→' if is_send else '←'
        action = 'sending' if is_send else 'receiving'
        phase = operation.split('_')[1]
        print(f"{operation} | {action} {phase} {pp_rank} "
              f"{direction} {peer_rank} | STEP:{STEP} | RANK:{pp_rank}", 
              flush=True)
    
    # Execute communication
    reqs = dist.batch_isend_irecv([op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    
    if VERBOSE:
        STEP += 1
    
    return tensor if not is_send else None


def bidirectional_pipeline_communicate(operation, 
                                       pp_group, 
                                       pp_rank, 
                                       send_tensor, 
                                       recv_shapes, 
                                       device, 
                                       dtype,
                                       is_first_stage=False,
                                       is_last_stage=False
                                       ):
    """
    Handles bidirectional communication between pipeline stages (send and recv simultaneously).
    Used in 1F1B schedule to overlap forward and backward communications.
    
    Args:
        operation: One of ['send_fwd_recv_bwd', 'send_bwd_recv_fwd']
        pp_group: Pipeline parallel process group
        pp_rank: Rank within the pipeline parallel group
        send_tensor: Tensor to send
        recv_shapes: Shape tuple for receiving tensor
        device: Device to allocate tensors on
        dtype: Data type for tensors
    
    Returns:
        Received tensor
    """
    global STEP
    global VERBOSE
    
    is_fwd = (operation == 'send_fwd_recv_bwd')
    
    # Boundary conditions: first/last stages don't do bidirectional communication
    if (is_fwd and pp_rank == dist.get_world_size(group=pp_group) - 1) or (not is_fwd and pp_rank == 0):
        return None
    
    # Determine peer rank based on operation
    peer_rank = pp_rank + 1 if is_fwd else pp_rank - 1
    
    # Allocate receive buffer
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    
    # Create bidirectional operations
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor, peer_rank, group=pp_group),
        dist.P2POp(dist.irecv, recv_tensor, peer_rank, group=pp_group)
    ])
    
    if VERBOSE:
        send_dir = 'next' if is_fwd else 'prev'
        recv_dir = 'next' if is_fwd else 'prev'
        print(f"{operation} | sending {send_dir} {pp_rank} -> {peer_rank} | "
              f"receiving {recv_dir} {peer_rank} -> {pp_rank} | "
              f"STEP={STEP} | RANK:{pp_rank}", flush=True)
    
    # Wait for both operations to complete
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    
    if VERBOSE:
        STEP += 1
    
    return recv_tensor
