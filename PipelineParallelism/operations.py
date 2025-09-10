import torch
import torch.distributed as dist
import torch.nn as nn


class Send(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dest_rank, group):
        """
        Send tensor to destination rank
        """
        print(f"Send.forward: Sending tensor {tensor.shape} to rank {dest_rank}")
        
        # Store context for backward pass
        ctx.dest_rank = dest_rank
        ctx.group = group
        ctx.tensor_shape = tensor.shape
        ctx.tensor_dtype = tensor.dtype
        
        # Create a copy of tensor to send (avoid modifying original)
        tensor_to_send = tensor.clone().detach()
        
        try:
            # CRITICAL: Make tensor contiguous before sending
            tensor_to_send = tensor_to_send.contiguous()
            
            # Send tensor synchronously
            print(f"Send.forward: About to call dist.send...")
            dist.send(tensor_to_send, dest_rank, group=group)
            print(f"Send.forward: dist.send completed successfully")
            
            # Return original tensor to maintain computation graph
            return tensor
            
        except Exception as e:
            print(f"Send.forward: ERROR during send: {e}")
            print(f"Send.forward: Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            print(f"Send.forward: Dest rank: {dest_rank}, Group: {group}")
            raise e

    @staticmethod
    def backward(ctx, grad_output):
        """
        Receive gradients from destination rank
        """
        print(f"Send.backward: Waiting for gradients from rank {ctx.dest_rank}")
        
        try:
            # Create tensor to receive gradients
            grad_tensor = torch.zeros(
                ctx.tensor_shape, 
                dtype=ctx.tensor_dtype,
                device=grad_output.device
            )
            
            print(f"Send.backward: About to call dist.recv for gradients...")
            # Receive gradients synchronously
            dist.recv(grad_tensor, ctx.dest_rank, group=ctx.group)
            print(f"Send.backward: Received gradients successfully")
            
            return grad_tensor, None, None
            
        except Exception as e:
            print(f"Send.backward: ERROR during gradient recv: {e}")
            raise e


class Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_rank, device, group, tensor_shape, tensor_dtype=torch.float32):
        """
        Receive tensor from source rank
        """
        print(f"Recv.forward: Waiting for tensor from rank {src_rank}")
        
        # Store context for backward pass
        ctx.src_rank = src_rank
        ctx.group = group
        print("tensor shape - ", tensor_shape)
        try:
            if tensor_shape is not None:
                # If shape is known, create tensor with correct shape
                received_tensor = torch.zeros(tensor_shape, dtype=tensor_dtype, device=device)
            else:
                # This is problematic - we need to know the shape ahead of time
                # For now, let's try a common approach: probe the message size
                print(f"Recv.forward: WARNING - No tensor shape provided, this might cause issues")
                # You might need to implement a shape exchange protocol here
                received_tensor = torch.empty(0, device=device)  # This will likely fail
            
            print(f"Recv.forward: About to call dist.recv...")
            print(f"Recv.forward: Expected tensor shape: {received_tensor.shape if tensor_shape else 'UNKNOWN'}")
            
            # Receive tensor synchronously
            dist.recv(received_tensor, src_rank, group=group)
            print(f"Recv.forward: dist.recv completed, received tensor shape: {received_tensor.shape}")
            
            # Store shape and dtype for backward pass
            ctx.tensor_shape = received_tensor.shape
            ctx.tensor_dtype = received_tensor.dtype
            
            return received_tensor
            
        except Exception as e:
            print(f"Recv.forward: ERROR during recv: {e}")
            print(f"Recv.forward: Src rank: {src_rank}, Device: {device}, Group: {group}")
            if tensor_shape:
                print(f"Recv.forward: Expected shape: {tensor_shape}, dtype: {tensor_dtype}")
            raise e

    @staticmethod
    def backward(ctx, grad_output):
        """
        Send gradients back to source rank
        """
        print(f"Recv.backward: Sending gradients back to rank {ctx.src_rank}")
        
        try:
            # Make gradient contiguous before sending
            grad_to_send = grad_output.clone().detach().contiguous()
            
            print(f"Recv.backward: About to call dist.send for gradients...")
            # Send gradients back synchronously
            dist.send(grad_to_send, ctx.src_rank, group=ctx.group)
            print(f"Recv.backward: Sent gradients successfully")
            
            return None, None, None, None, None
            
        except Exception as e:
            print(f"Recv.backward: ERROR during gradient send: {e}")
            raise e