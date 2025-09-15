# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import os

# class Send(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, tensor, dest_rank, group):
#         """
#         Send a tensor to the destination rank.
#         Protocol:
#         1. Send the number of dimensions (1-element tensor).
#         2. Send the shape (N-element tensor, where N is the number of dimensions).
#         3. Send the actual tensor data.
#         """
#         # Ensure tensor is on the correct device and contiguous
#         tensor_to_send = tensor.contiguous()

#         # Store context for the backward pass
#         ctx.dest_rank = dest_rank
#         ctx.group = group
#         ctx.tensor_shape = tensor.shape
#         ctx.tensor_dtype = tensor.dtype

#         # 1. Send the number of dimensions
#         # FIX: Used tensor.dim() for a more direct way to get the number of dimensions.
#         num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
#         dist.send(tensor=num_dims, dst=dest_rank, group=group)

#         # 2. Send the shape
#         shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
#         dist.send(tensor=shape_tensor, dst=dest_rank, group=group)

#         # 3. Send the tensor data
#         dist.send(tensor=tensor_to_send, dst=dest_rank, group=group)
        
#         # Return the original tensor to maintain the computational graph on the sender's side
#         return tensor

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Receive gradients from the destination rank during the backward pass.
#         """
#         # The gradient is coming from the rank we sent the tensor to.
#         src_rank = ctx.dest_rank
        
#         # Pre-allocate a tensor to receive the gradient
#         grad_tensor = torch.zeros(
#             ctx.tensor_shape,
#             dtype=ctx.tensor_dtype,
#             device=grad_output.device  # Ensure grad is on the same device
#         )
        
#         dist.recv(tensor=grad_tensor, src=src_rank, group=ctx.group)
        
#         # The returned gradients must correspond to the inputs of the forward function.
#         # In this case: tensor, dest_rank, group
#         return grad_tensor, None, None


# class Recv(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, src_rank, device, group):
#         """
#         Receive a tensor from the source rank.
#         Protocol matches Send.forward:
#         1. Receive the number of dimensions.
#         2. Receive the shape.
#         3. Receive the actual tensor data.
#         """
#         # Store context for the backward pass
#         ctx.src_rank = src_rank
#         ctx.group = group
#         dtype = torch.float32  # Default dtype; can be modified as needed
#         # 1. Receive the number of dimensions
#         num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
#         # FIX: Changed src=0 to src=src_rank to receive from the correct source.
#         dist.recv(tensor=num_dims_tensor, src=src_rank, group=group)
#         num_dims = num_dims_tensor.item()

#         # 2. Receive the shape
#         shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
#         # FIX: Changed src=0 to src=src_rank here as well.
#         dist.recv(tensor=shape_tensor, src=src_rank, group=group)
#         tensor_shape = shape_tensor.tolist()
        
#         # 3. Receive the tensor data
#         # FIX: The print statement no longer causes a NameError.
#         # print(f"Recv.forward: Preparing to receive tensor with shape {tensor_shape}")
#         received_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
#         dist.recv(tensor=received_tensor, src=src_rank, group=group)

#         return received_tensor

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Send gradients back to the source rank during the backward pass.
#         """
#         # Send the gradient to the rank we received the tensor from.
#         dest_rank = ctx.src_rank
        
#         # Ensure gradient is contiguous before sending
#         grad_to_send = grad_output.contiguous()
#         dist.send(tensor=grad_to_send, dst=dest_rank, group=ctx.group)
        
#         # The returned gradients must correspond to the inputs of the forward function.
#         # Inputs: src_rank, device, group, dtype
#         # FIX: Returned 4 None values to match the 4 inputs of the forward function.
#         return None, None, None, None


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
        print(f"Send.forward: rank {dist.get_rank(group)} sending to rank {dest_rank}")
        print(f"Send.forward: tensor shape {tensor.shape}, device {tensor.device}, dtype {tensor.dtype}")
        
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
            print(f"Send.forward: sending num_dims {num_dims.item()}")
            dist.send(tensor=num_dims, dst=dest_rank, group=group)

            # 2. Send the shape
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
            print(f"Send.forward: sending shape {shape_tensor.tolist()}")
            dist.send(tensor=shape_tensor, dst=dest_rank, group=group)

            # 3. Send the tensor data
            print(f"Send.forward: sending tensor data...")
            dist.send(tensor=tensor_to_send, dst=dest_rank, group=group)
            print(f"Send.forward: tensor data sent successfully")
            
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
        print(f"Send.backward: rank {dist.get_rank(ctx.group)} waiting for grad from rank {ctx.dest_rank}")
        
        # Pre-allocate a tensor to receive the gradient
        grad_tensor = torch.zeros(
            ctx.tensor_shape,
            dtype=ctx.tensor_dtype,
            device=grad_output.device
        )
        
        try:
            print(f"Send.backward: receiving gradient tensor...")
            dist.recv(tensor=grad_tensor, src=ctx.dest_rank, group=ctx.group)
            print(f"Send.backward: gradient received successfully, norm: {grad_tensor.norm().item()}")
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
        print(f"Recv.forward: rank {dist.get_rank(group)} receiving from rank {src_rank}")
        
        # Store context for the backward pass
        ctx.src_rank = src_rank
        ctx.group = group
        ctx.dtype = dtype

        try:
            # 1. Receive the number of dimensions
            num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
            print(f"Recv.forward: waiting for num_dims...")
            dist.recv(tensor=num_dims_tensor, src=src_rank, group=group)
            num_dims = num_dims_tensor.item()
            print(f"Recv.forward: received num_dims: {num_dims}")

            # 2. Receive the shape
            shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
            print(f"Recv.forward: waiting for shape...")
            dist.recv(tensor=shape_tensor, src=src_rank, group=group)
            tensor_shape = shape_tensor.tolist()
            print(f"Recv.forward: received shape: {tensor_shape}")
            
            # 3. Receive the tensor data
            print(f"Recv.forward: waiting for tensor data...")
            received_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
            dist.recv(tensor=received_tensor, src=src_rank, group=group)
            print(f"Recv.forward: received tensor successfully, norm: {received_tensor.norm().item()}")
            
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
        print(f"Recv.backward: rank {dist.get_rank(ctx.group)} sending grad to rank {ctx.src_rank}")
        print(f"Recv.backward: grad_output norm: {grad_output.norm().item()}")
        
        # Send the gradient to the rank we received the tensor from
        dest_rank = ctx.src_rank
        
        try:
            # Ensure gradient is contiguous before sending
            grad_to_send = grad_output.contiguous()
            dist.send(tensor=grad_to_send, dst=dest_rank, group=ctx.group)
            print(f"Recv.backward: gradient sent successfully")
        except Exception as e:
            print(f"Recv.backward: ERROR - {e}")
            raise
        
        return None, None, None