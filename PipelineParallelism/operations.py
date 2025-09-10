# corrected_pipeline.py (minimal, self-contained stage + wrappers)
import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn

# It's better to use a larger, more flexible header if you anticipate
# tensors with many dimensions. Using a single tensor for metadata is also cleaner.
# [num_dims, dim1, dim2, ...]
# We will use a more direct method by sending the shape tensor itself.

class Send(Function):
    """
    Autograd-friendly send: forward sends a tensor to `dst`.
    Backward receives the gradient tensor from `dst`.
    """
    @staticmethod
    def forward(ctx, tensor, dst, group=None):
        ctx.dst = dst
        ctx.group = group
        ctx.tensor_shape = tuple(tensor.shape)
        ctx.tensor_dtype = tensor.dtype
        ctx.tensor_device = tensor.device

        # Send metadata (shape and dtype) first
        metadata = torch.tensor(list(ctx.tensor_shape), dtype=torch.long, device=tensor.device)
        dist.send(metadata, dst=ctx.dst, group=ctx.group)
        
        # Send the actual tensor
        dist.send(tensor.contiguous(), dst=ctx.dst, group=ctx.group)
        
        # Return an empty tensor as sender doesn't need the output
        return torch.empty(0, device=tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is an empty tensor
        # Receive the gradient from the destination rank
        grad = torch.empty(ctx.tensor_shape, dtype=ctx.tensor_dtype, device=ctx.tensor_device)
        dist.recv(grad, src=ctx.dst, group=ctx.group)
        
        # Return gradient for the input tensor and None for other args
        return grad, None, None


class Recv(Function):
    """
    Autograd-friendly recv: forward receives a tensor from `src`.
    Backward sends the gradient back to `src`.
    """
    @staticmethod
    def forward(ctx, src, device, group=None):
        ctx.src = src
        ctx.group = group
        
        # Receive metadata (shape) first
        # Assuming a reasonable max number of dimensions (e.g., 16)
        metadata = torch.zeros(16, dtype=torch.long, device=device)
        dist.recv(metadata, src=ctx.src, group=ctx.group)
        
        # The first element of metadata could be num_dims, or we can filter out zeros
        shape = tuple(dim.item() for dim in metadata if dim.item() != 0)
        
        # Receive the tensor with the correct shape
        tensor = torch.empty(shape, device=device) # Assuming dtype is torch.float32 for now
        dist.recv(tensor, src=ctx.src, group=ctx.group)

        ctx.tensor_shape = tensor.shape
        
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Send the gradient back to the source rank
        dist.send(grad_output.contiguous(), dst=ctx.src, group=ctx.group)
        
        # Return None for the inputs to forward (src, device, group)
        return None, None, None