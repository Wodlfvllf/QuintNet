import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper
from .operations import Send, Recv


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
from tqdm import tqdm
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

    def train_step_naive(self, input_data, target):
        self.model.train()
        self.optimizer.zero_grad()

        # --- 1. FORWARD PASS CHAIN ---
        if self.is_first_stage:
            self.output_tensor_of_stage = self.model(input_data.to(self.device))
            # MODIFIED: Use the helper to send the tensor and its metadata.
            send_tensor_with_header(self.output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)
        
        else: # For last and intermediate stages
            # MODIFIED: Use the helper to receive. It discovers the shape automatically.
            buffer = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=input_data.dtype)
            
            self.input_tensor_for_stage = buffer
            self.input_tensor_for_stage.requires_grad = True
            
            self.output_tensor_of_stage = self.model(self.input_tensor_for_stage)
            
            if not self.is_last_stage:
                send_tensor_with_header(self.output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)

        # --- 2. BACKWARD PASS CHAIN (in reverse) ---
        if self.is_last_stage:
            loss = self.criterion(self.output_tensor_of_stage, target.to(device=self.device, dtype=torch.long))
            loss.backward()
            
            # MODIFIED: Use the helper to send the gradient and its metadata.
            grad_to_send = self.input_tensor_for_stage.grad
            send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
            
        else: # For first and intermediate stages
            # MODIFIED: Use the helper to receive the gradient.
            # We know the gradient will have the same shape and dtype as our stage's output.
            grad_buffer = recv_tensor_with_header(self.rank + 1, self.device, self.pp_group, dtype=self.output_tensor_of_stage.dtype)
            
            self.output_tensor_of_stage.backward(gradient=grad_buffer)
            
            if not self.is_first_stage:
                grad_to_send = self.input_tensor_for_stage.grad
                send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
        
        # --- 3. OPTIMIZER STEP ---
        self.optimizer.step()
        
        if self.is_last_stage:
            acc = (self.output_tensor_of_stage.argmax(dim=1) == target.to(self.device)).float().mean().item()
            return loss.item(), acc * 100
        else:
            return None, None
        
    # def train_all_forward_and_backward_optimised(self, data_loader, pgm, requires_grad_sync=True):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     input_tensors = []
    #     output_tensors = []
    #     logging_loss = 0.0
    #     device = self.device
    #     acc = 0.0
        
    #     # Convert data_loader to list to allow multiple iterations
    #     batches = list(data_loader)
        
    #     # All forward passes
    #     for i, batch in enumerate(tqdm(batches)):
    #         if self.is_first_stage:
    #             output_tensor = self.model(batch['image'].to(device))
    #             send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
    #             input_tensor = None
    #         else:
    #             input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
    #             output_tensor = self.model(input_tensor)
    #         input_tensors.append(input_tensor)
    #         output_tensors.append(output_tensor)

    #     # All backward passes (in reverse order)
    #     for i in range(len(batches) - 1, -1, -1):
    #         batch = batches[i]
            
    #         if self.is_last_stage:
    #             # Get target labels based on batch structure
    #             if "labels" in batch:
    #                 target = batch["labels"].to(device, dtype=torch.long)
    #             else:
    #                 target = batch["label"].to(device, dtype=torch.long)
                
    #             output_tensor = output_tensors.pop()
    #             loss = self.criterion(output_tensor, target)
    #             logging_loss += loss.item()
    #             loss.backward()
    #             acc = (self.output_tensor_of_stage.argmax(dim=1) == target.to(self.device)).float().mean().item()
                
    #             if not self.is_first_stage:
    #                 grad_to_send = input_tensors.pop().grad
    #                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
    #             else:
    #                 input_tensors.pop()  # Remove from list even if not sending
    #         else:
    #             # Receive gradient from next stage
    #             grad_buffer = recv_tensor_with_header(self.rank + 1, device, self.pp_group, dtype=dtype)
    #             output_tensor = output_tensors.pop()
    #             output_tensor.backward(gradient=grad_buffer)
                
    #             if not self.is_first_stage:
    #                 grad_to_send = input_tensors.pop().grad
    #                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
    #             else:
    #                 input_tensors.pop()  # Remove from list even if not sending

    #     # Synchronize gradients across all stages
    #     if requires_grad_sync:
    #         dist.barrier(group=self.pp_group)
        
    #     # Optimizer step
    #     self.optimizer.step()
        
    #     # Correct syntax
    #     return logging_loss / len(batches), (acc if self.is_last_stage else 0.0), 0.0
    
    def train_all_forward_and_backward_optimised(self, data_loader, pgm, requires_grad_sync=True):
        self.model.train()
        self.optimizer.zero_grad()
        input_tensors = []
        output_tensors = []
        logging_loss = 0.0
        device = self.device
        acc = 0.0
        
        # Convert data_loader to list to allow multiple iterations
        batches = list(data_loader)
        
        # All forward passes
        for i, batch in enumerate(tqdm(batches)):
            if self.is_first_stage:
                output_tensor = self.model(batch['image'].to(device))
                send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                input_tensor = None # First stage has no input tensor from another stage
                
                # Store the model's direct input for the backward pass
                # This requires saving the original input `batch['image']`
                # and making it require grad if it's the very first stage
                # This part of the logic seems missing in the original code but is crucial.
                # For simplicity, we continue storing `None` as per the original code's structure.
                
            else:
                # THIS IS THE FIX: Receive from the PREVIOUS stage
                input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
                input_tensor.requires_grad = True # Make sure to set this for the backward pass
                output_tensor = self.model(input_tensor)
                
                if not self.is_last_stage:
                    # If not the last stage, send to the next one
                    send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        # All backward passes (in reverse order)
        for i in range(len(batches) - 1, -1, -1):
            batch = batches[i]
            
            # Pop the corresponding tensors for this micro-batch
            input_tensor_for_grad = input_tensors.pop()
            output_tensor = output_tensors.pop()
            
            if self.is_last_stage:
                # Get target labels based on batch structure
                target = batch.get("labels", batch.get("label")).to(device, dtype=torch.long)
                
                loss = self.criterion(output_tensor, target)
                logging_loss += loss.item()
                loss.backward()
                
                # Use the correct output tensor for accuracy calculation
                acc += (output_tensor.argmax(dim=1) == target).float().mean().item()
                
                if input_tensor_for_grad is not None:
                    grad_to_send = input_tensor_for_grad.grad
                    send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
            else:
                # Receive gradient from next stage
                # The dtype should match the output tensor's dtype
                grad_buffer = recv_tensor_with_header(self.rank + 1, device, self.pp_group, dtype=output_tensor.dtype)
                output_tensor.backward(gradient=grad_buffer)
                
                if not self.is_first_stage and input_tensor_for_grad is not None:
                    grad_to_send = input_tensor_for_grad.grad
                    send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)

        # Synchronize gradients across all stages
        if requires_grad_sync:
            dist.barrier(group=self.pp_group)
        
        # Optimizer step
        self.optimizer.step()
        
        # Calculate average accuracy for the last stage
        if self.is_last_stage:
            acc /= len(batches)

        # Return the average loss and accuracy
        return logging_loss / len(batches), acc if self.is_last_stage else 0.0
            
    def evaluate(self, val_loader):
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
                        # MODIFIED: Use the helper for evaluation as well.
                        send_tensor_with_header(output, self.rank + 1, self.pp_group)
                else:
                    # MODIFIED: The helper removes the need for get_input_shape.
                    buffer = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=images.dtype)
                    output = self.model(buffer)
                    if not self.is_last_stage:
                        send_tensor_with_header(output, self.rank + 1, self.pp_group)

                if self.is_last_stage:
                    targets = targets.to(self.device, dtype=torch.long)
                    loss = self.criterion(output, targets)
                    total_loss += loss.item() * output.size(0)
                    total_correct += (output.argmax(dim=1) == targets).sum().item()
                    total_samples += output.size(0)

        dist.barrier(group=self.pp_group)
        
        if self.is_last_stage:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return None, None
        
        
