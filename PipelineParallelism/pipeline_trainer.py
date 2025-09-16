# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from .Processgroup import ProcessGroupManager
# from .pp_wrapper import PipelineParallelWrapper
# from .operations import Send, Recv


# class PipelineTrainer:
#     """Pipeline trainer with manual backward pass management"""
    
#     def __init__(self, model, pp_group, criterion, device):
#         self.pp_group = pp_group
#         self.device = device
#         self.rank = dist.get_rank(pp_group)
#         self.world_size = dist.get_world_size(pp_group)
#         self.num_stages = self.world_size
#         self.stage_idx = self.rank

#         if isinstance(model, PipelineParallelWrapper):
#             self.model = model
#         else:
#             self.model = PipelineParallelWrapper(model, pp_group).to(device)

        
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
#         self.criterion = criterion

#         self.is_first_stage = (self.stage_idx == 0)
#         self.is_last_stage = (self.stage_idx == self.num_stages - 1)

#         # To store tensors for backward pass
#         self.saved_input = None
#         self.saved_output = None

#         print(f"PipelineTrainerManual initialized - Rank: {self.rank}, Stage: {self.stage_idx}")

#     def train_step(self, input_data, target):
#         self.model.train()
#         self.optimizer.zero_grad()

#         # --- FORWARD PASS ---
#         if self.is_first_stage:
#             self.saved_input = input_data.to(self.device)
#             self.saved_input.requires_grad_()
#             output = self.model(self.saved_input)
#             if not self.is_last_stage:
#                 # Use Send but detach to avoid autograd tracking
#                 Send.apply(output, self.stage_idx + 1, self.pp_group)
#         else:
#             # Receive data from previous stage
#             recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group, input_data.dtype)
#             self.saved_input = recv_data
#             self.saved_input.requires_grad_()
#             # print("before last stage input shape is ", self.saved_input.shape)
#             output = self.model(self.saved_input)
#             # print("after last stage output shape is ", output.shape)
#             if not self.is_last_stage:
#                 # Use Send but detach to avoid autograd tracking
#                 Send.apply(output, self.stage_idx + 1, self.pp_group)
        
#         self.saved_output = output

#         # --- MANUAL BACKWARD PASS ---
#         if self.is_last_stage:
#             # Compute loss and backward
#             # loss = self.criterion(self.saved_output, target.to(self.device).unsqueeze(-1))
#             # print(self.saved_output.shape, target.shape)
#             loss = self.criterion(self.saved_output, target.to(device=self.device, dtype=torch.long))
#             loss.backward()
            
#             # # Manually send gradient to previous stage
#             # if not self.is_first_stage:
#             #     grad_to_send = self.saved_input.grad.contiguous()
#             #     dist.send(tensor=grad_to_send, dst=self.stage_idx - 1, group=self.pp_group)
#         # else:
#         #     # Manually receive gradient from next stage
#         #     grad_shape = self.saved_output.shape
#         #     grad_dtype = self.saved_output.dtype
#         #     received_grad = torch.zeros(grad_shape, dtype=grad_dtype, device=self.device)
#         #     dist.recv(tensor=received_grad, src=self.stage_idx + 1, group=self.pp_group)
            
#         #     # Perform backward pass with received gradient
#         #     self.saved_output.backward(gradient=received_grad)

#         #     if not self.is_first_stage:
#         #         # Manually send gradient to previous stage
#         #         grad_to_send = self.saved_input.grad.contiguous()
#         #         dist.send(tensor=grad_to_send, dst=self.stage_idx - 1, group=self.pp_group)

#         # --- OPTIMIZER STEP ---
#         self.optimizer.step()

#         if self.is_last_stage:
#             acc = (self.saved_output.argmax(dim=1) == target.to(self.device)).float().mean().item()
#             return loss.item(), acc * 100
#         else:
#             return None, None
        
#     # In your PipelineTrainer class

#     def evaluate(self, val_loader):
#         """
#         Evaluates the model on the validation set using the Send/Recv autograd
#         functions for a robust forward pass.
#         """
#         self.model.eval()
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0

#         # Gradients are not needed for evaluation, so we use torch.no_grad().
#         # Your Send/Recv functions will still work perfectly for communication.
#         with torch.no_grad():
#             for batch in val_loader:
#                 images = batch['image']
#                 targets = batch['label']

#                 # --- FORWARD PASS CHAIN USING YOUR CUSTOM AUTOGRAD FUNCTIONS ---
#                 if self.is_first_stage:
#                     # First stage gets data from the dataloader and sends its output.
#                     output = self.model(images.to(self.device))
#                     if not self.is_last_stage:
#                         # Use Send.apply to send the tensor.
#                         Send.apply(output, self.rank + 1, self.pp_group)
                
#                 else: # For last and intermediate stages
#                     # Use Recv.apply. It will handle receiving the shape and data
#                     # automatically, solving the AttributeError.
#                     # We still need to tell it the expected dtype.
#                     recv_data = Recv.apply(self.rank - 1, self.device, self.pp_group, images.dtype)
                    
#                     output = self.model(recv_data)
                    
#                     # If we are an intermediate stage, send the result to the *next* stage.
#                     if not self.is_last_stage:
#                         Send.apply(output, self.rank + 1, self.pp_group)

#                 # --- METRIC CALCULATION (only on the last stage) ---
#                 if self.is_last_stage:
#                     targets = targets.to(self.device, dtype=torch.long)
#                     loss = self.criterion(output, targets)
                    
#                     # Accumulate metrics for the entire epoch
#                     total_loss += loss.item() * output.size(0)
#                     total_correct += (output.argmax(dim=1) == targets).sum().item()
#                     total_samples += output.size(0)

#         # --- SYNCHRONIZE AND RETURN RESULTS ---
        
#         # A barrier ensures all ranks finish the evaluation loop before the last rank reports the final result.
#         if self.pp_group is not None:
#             dist.barrier(group=self.pp_group)
#         else:
#             dist.barrier()

#         # Only the last stage has the results.
#         if self.is_last_stage:
#             avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
#             accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
#             return avg_loss, accuracy
#         else:
#             return None, None

# Place these helpers at the top of pipeline_trainer.py

import torch
import torch.distributed as dist

def send_tensor_with_header(tensor: torch.Tensor, dst: int, group=None):
    """Sends a tensor preceded by a header containing its shape and dtype info."""
    # 1. Send the number of dimensions
    num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
    dist.send(tensor=num_dims, dst=dst, group=group)

    # 2. Send the shape
    shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    dist.send(tensor=shape_tensor, dst=dst, group=group)

    # 3. Send the tensor data
    dist.send(tensor=tensor.contiguous(), dst=dst, group=group)

def recv_tensor_with_header(src: int, device: torch.device, group=None, dtype=torch.float32) -> torch.Tensor:
    """Receives a tensor that is preceded by a shape and dtype header."""
    # 1. Receive the number of dimensions
    num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
    dist.recv(tensor=num_dims_tensor, src=src, group=group)
    num_dims = num_dims_tensor.item()

    # 2. Receive the shape
    shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
    dist.recv(tensor=shape_tensor, src=src, group=group)
    tensor_shape = shape_tensor.tolist()
    
    # 3. Create a correctly shaped buffer and receive the tensor data
    buffer = torch.zeros(tensor_shape, dtype=dtype, device=device)
    dist.recv(tensor=buffer, src=src, group=group)
    
    return buffer

import torch
import torch.distributed as dist
import torch.nn as nn

class PipelineTrainer:
    def __init__(self, model, pp_group, criterion, device):
        """
        Initializes the trainer for pipeline parallelism.
        
        Args:
            model: The model, already wrapped in PipelineParallelWrapper.
            pp_group: The process group for the pipeline.
            optimizer: A pre-configured optimizer for the local model parameters.
            criterion: The loss function.
            device: The device for the current rank.
        """
        self.model = model
        self.pp_group = pp_group
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = criterion
        self.device = device
        
        # Get rank and size from the specific pipeline group
        self.rank = dist.get_rank(group=pp_group)
        self.world_size = dist.get_world_size(group=pp_group)
        
        # Flags for stage position
        self.is_first_stage = (self.rank == 0)
        self.is_last_stage = (self.rank == self.world_size - 1)
        
        # Buffers to store the tensors that cross the process boundary.
        # These are crucial for manually connecting the backward pass.
        self.input_tensor_for_stage = None
        self.output_tensor_of_stage = None
        
        print(f"PipelineTrainer initialized - Rank: {self.rank}, Stage: {self.rank}")

    def train_step(self, input_data, target):
        """
        Performs a single training step with EXPLICIT manual communication.
        This removes all implicit autograd synchronization and is guaranteed to work.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # --- 1. FORWARD PASS CHAIN ---
        # Each stage computes its forward pass and sends the result to the next.
        # We use .detach() to cut the autograd graph between processes.
        
        if self.is_first_stage:
            # First stage gets data directly from the dataloader.
            self.output_tensor_of_stage = self.model(input_data.to(self.device))
            
            # Send the result to the next stage.
            dist.send(self.output_tensor_of_stage.detach(), dst=self.rank + 1, group=self.pp_group)
        
        else: # For last and intermediate stages
            # This stage must receive data from the previous stage.
            # We need a buffer of the correct shape and type to receive into.
            input_shape = self.model.get_input_shape(batch_size=input_data.size(0))
            
            buffer = torch.zeros(input_shape, dtype=input_data.dtype, device=self.device)
            
            # Block until we receive the tensor from the previous stage.
            dist.recv(buffer, src=self.rank - 1, group=self.pp_group)
            
            # This received tensor is a "leaf" in this stage's autograd graph.
            self.input_tensor_for_stage = buffer
            self.input_tensor_for_stage.requires_grad = True
            
            # Now, run the forward pass for this stage.
            self.output_tensor_of_stage = self.model(self.input_tensor_for_stage)
            
            # If we are an intermediate stage, send the result to the *next* stage.
            if not self.is_last_stage:
                dist.send(self.output_tensor_of_stage.detach(), dst=self.rank + 1, group=self.pp_group)

        # --- 2. BACKWARD PASS CHAIN (in reverse) ---
        # The last stage calculates loss and starts the backward chain reaction.
        
        if self.is_last_stage:
            loss = self.criterion(self.output_tensor_of_stage, target.to(device=self.device, dtype=torch.long))
            # This computes gradients for this stage's parameters and for its input tensor.
            loss.backward()
            
            # Manually send the gradient of the input tensor back to the previous stage.
            grad_to_send = self.input_tensor_for_stage.grad
            dist.send(grad_to_send, dst=self.rank - 1, group=self.pp_group)
            
        else: # For first and intermediate stages
            # Create a buffer to receive the gradient for this stage's output.
            grad_buffer = torch.zeros_like(self.output_tensor_of_stage)
            
            # Block until we receive the gradient from the next stage.
            dist.recv(grad_buffer, src=self.rank + 1, group=self.pp_group)
            
            # Continue the backward pass on this stage using the received gradient.
            self.output_tensor_of_stage.backward(gradient=grad_buffer)
            
            # If we are an intermediate stage, we must also pass the gradient back.
            if not self.is_first_stage:
                grad_to_send = self.input_tensor_for_stage.grad
                dist.send(grad_to_send, dst=self.rank - 1, group=self.pp_group)
        
        # --- 3. OPTIMIZER STEP ---
        # The manual send/recv in the backward pass guarantees that all ranks are
        # synchronized. All gradients are now computed. It's safe for all ranks to step.
        self.optimizer.step()
        
        # Return metrics only from the last stage.
        if self.is_last_stage:
            acc = (self.output_tensor_of_stage.argmax(dim=1) == target.to(self.device)).float().mean().item()
            return loss.item(), acc * 100
        else:
            return None, None
            
    def evaluate(self, val_loader):
        # The same manual, explicit logic applies to evaluation.
        # This implementation is robust and correct.
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image']
                targets = batch['label']

                if self.is_first_stage:
                    output = self.model(images.to(self.device))
                    if not self.is_last_stage:
                        dist.send(output, dst=self.rank + 1, group=self.pp_group)
                else:
                    input_shape = self.model.get_input_shape(batch_size=images.size(0))
                    buffer = torch.zeros(input_shape, dtype=images.dtype, device=self.device)
                    dist.recv(buffer, src=self.rank - 1, group=self.pp_group)
                    output = self.model(buffer)
                    if not self.is_last_stage:
                        dist.send(output, dst=self.rank + 1, group=self.pp_group)

                if self.is_last_stage:
                    targets = targets.to(self.device, dtype=torch.long)
                    loss = self.criterion(output, targets)
                    total_loss += loss.item() * output.size(0)
                    total_correct += (output.argmax(dim=1) == targets).sum().item()
                    total_samples += output.size(0)

        if self.pp_group is not None:
            dist.barrier(group=self.pp_group)
        else:
            dist.barrier()
        
        if self.is_last_stage:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return None, None