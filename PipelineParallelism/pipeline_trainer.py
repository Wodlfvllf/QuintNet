import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper
from .operations import Send, Recv
import torch
import torch.distributed as dist
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from collections import deque
from tqdm import tqdm
from typing import Optional, Tuple

def send_tensor_with_header(tensor: torch.Tensor, dst: int, group=None):
    """Sends a tensor preceded by a header containing its shape and dtype info."""
    if tensor is None:
        # Send a flag indicating None
        flag = torch.tensor([-1], dtype=torch.long, device='cuda')
        dist.send(tensor=flag, dst=dst, group=group)
        return
    
    # Send flag indicating valid tensor
    flag = torch.tensor([1], dtype=torch.long, device=tensor.device)
    dist.send(tensor=flag, dst=dst, group=group)
    
    # Send the number of dimensions
    num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
    dist.send(tensor=num_dims, dst=dst, group=group)

    # Send the shape
    shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    dist.send(tensor=shape_tensor, dst=dst, group=group)

    # Send the tensor data
    dist.send(tensor=tensor.contiguous(), dst=dst, group=group)

def recv_tensor_with_header(src: int, device: torch.device, group=None, dtype=torch.float32) -> Optional[torch.Tensor]:
    """Receives a tensor that is preceded by a shape and dtype header."""
    # Receive flag
    flag = torch.zeros(1, dtype=torch.long, device=device)
    dist.recv(tensor=flag, src=src, group=group)
    
    if flag.item() == -1:
        return None
    
    # Receive the number of dimensions
    num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
    dist.recv(tensor=num_dims_tensor, src=src, group=group)
    num_dims = num_dims_tensor.item()

    # Receive the shape
    shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
    dist.recv(tensor=shape_tensor, src=src, group=group)
    tensor_shape = shape_tensor.tolist()
    
    # Create a correctly shaped buffer and receive the tensor data
    buffer = torch.zeros(tensor_shape, dtype=dtype, device=device)
    dist.recv(tensor=buffer, src=src, group=group)
    
    return buffer

import torch
import torch.distributed as dist
import torch.nn as nn
from collections import deque

class PipelineTrainer:
    def __init__(self, model, pp_group, criterion, device, optimizer=None, max_grad_norm=1.0):
        """
        Initializes the trainer for pipeline parallelism.
        
        Args:
            model: The model, already wrapped in PipelineParallelWrapper.
            pp_group: The process group for the pipeline.
            criterion: The loss function.
            device: The device for the current rank.
            optimizer: Optional pre-configured optimizer for the local model parameters.
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        self.model = model
        self.pp_group = pp_group
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # Get rank and size from the specific pipeline group
        self.rank = dist.get_rank(group=pp_group)
        self.world_size = dist.get_world_size(group=pp_group)
        
        # Flags for stage position
        self.is_first_stage = (self.rank == 0)
        self.is_last_stage = (self.rank == self.world_size - 1)
        
        print(f"PipelineTrainer initialized - Rank: {self.rank}, Stage: {self.rank}/{self.world_size-1}")
        print(f"  First stage: {self.is_first_stage}, Last stage: {self.is_last_stage}")
        print(f"  Device: {device}, Max grad norm: {max_grad_norm}")

    def train_step_naive(self, input_data, target):
        """
        Simple training step - one forward, one backward, one update.
        Good for debugging but not efficient.
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_tensor_for_stage = None
        output_tensor_of_stage = None

        # --- FORWARD PASS ---
        if self.is_first_stage:
            input_data = input_data.to(self.device)
            output_tensor_of_stage = self.model(input_data)
            
            if not self.is_last_stage:
                send_tensor_with_header(output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)
        else:
            # Receive from previous stage
            input_tensor_for_stage = recv_tensor_with_header(
                self.rank - 1, self.device, self.pp_group, dtype=input_data.dtype
            )
            input_tensor_for_stage.requires_grad = True
            
            output_tensor_of_stage = self.model(input_tensor_for_stage)
            
            if not self.is_last_stage:
                send_tensor_with_header(output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)

        # --- BACKWARD PASS ---
        if self.is_last_stage:
            target = target.to(device=self.device, dtype=torch.long)
            loss = self.criterion(output_tensor_of_stage, target)
            loss.backward()
            
            if not self.is_first_stage and input_tensor_for_stage is not None:
                grad_to_send = input_tensor_for_stage.grad
                send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
        else:
            # Receive gradient from next stage
            grad_buffer = recv_tensor_with_header(
                self.rank + 1, self.device, self.pp_group, dtype=output_tensor_of_stage.dtype
            )
            
            output_tensor_of_stage.backward(gradient=grad_buffer)
            
            if not self.is_first_stage and input_tensor_for_stage is not None:
                grad_to_send = input_tensor_for_stage.grad
                send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
        
        # --- OPTIMIZER STEP ---
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Return loss and accuracy for last stage
        if self.is_last_stage:
            with torch.no_grad():
                acc = (output_tensor_of_stage.argmax(dim=1) == target).float().mean().item()
            return loss.item(), acc * 100
        else:
            return None, None
    
    def train_all_forward_and_backward_optimised(self, data_loader, pgm, epoch, requires_grad_sync=True):
        self.model.train()
        self.optimizer.zero_grad()
        input_tensors = []
        output_tensors = []
        logging_loss = 0.0
        device = self.device
        acc = 0.0
        
        # Convert data_loader to list to allow multiple iterations
        batches = list(data_loader)
        num_batches = len(batches)
        
        # All forward passes
        for i, batch in enumerate(tqdm(batches)):
            if self.is_first_stage:
                # FIXED: Store the actual input tensor for gradient computation
                input_data = batch['image'].to(device)
                input_data.requires_grad = True  # Critical for gradient flow
                output_tensor = self.model(input_data)
                
                if not self.is_last_stage:  # Only send if not last stage
                    send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                
                # Store the input data (not None) for backward pass
                input_tensor = input_data
                
            else:
                # Receive from the PREVIOUS stage
                input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
                input_tensor.requires_grad = True  # Make sure to set this for the backward pass
                output_tensor = self.model(input_tensor)
                
                if not self.is_last_stage:
                    # If not the last stage, send to the next one
                    send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

        # All backward passes (in reverse order to match LIFO stack behavior)
        for i in range(len(batches) - 1, -1, -1):
            batch = batches[i]
            
            # FIXED: Use pop() to get LIFO order (last stored, first retrieved)
            input_tensor_for_grad = input_tensors.pop()
            output_tensor = output_tensors.pop()
            
            if self.is_last_stage:
                # FIXED: Handle both possible label keys
                if 'labels' in batch:
                    target = batch['labels'].to(device, dtype=torch.long)
                elif 'label' in batch:
                    target = batch['label'].to(device, dtype=torch.long)
                else:
                    raise KeyError("Neither 'labels' nor 'label' found in batch")
                
                # FIXED: Scale loss by number of micro-batches for proper gradient averaging
                loss = self.criterion(output_tensor, target)
                scaled_loss = loss / num_batches  # Average the gradients
                logging_loss += loss.item()  # Log unscaled loss
                
                scaled_loss.backward()
                
                # Calculate accuracy
                with torch.no_grad():
                    acc += (output_tensor.argmax(dim=1) == target).float().mean().item()
                
                # Send gradient to previous stage (if not first stage)
                if not self.is_first_stage and input_tensor_for_grad is not None:
                    if input_tensor_for_grad.grad is not None:
                        send_tensor_with_header(input_tensor_for_grad.grad, self.rank - 1, self.pp_group)
                    else:
                        # Send zero gradient if no gradient computed
                        zero_grad = torch.zeros_like(input_tensor_for_grad)
                        send_tensor_with_header(zero_grad, self.rank - 1, self.pp_group)
                        
            else:
                # Receive gradient from next stage
                grad_buffer = recv_tensor_with_header(self.rank + 1, device, self.pp_group, dtype=output_tensor.dtype)
                
                # FIXED: Scale the received gradient by number of micro-batches
                scaled_grad_buffer = grad_buffer / num_batches
                output_tensor.backward(gradient=scaled_grad_buffer)
                
                # Send gradient to previous stage (if not first stage)
                if not self.is_first_stage and input_tensor_for_grad is not None:
                    if input_tensor_for_grad.grad is not None:
                        send_tensor_with_header(input_tensor_for_grad.grad, self.rank - 1, self.pp_group)
                    else:
                        # Send zero gradient if no gradient computed
                        zero_grad = torch.zeros_like(input_tensor_for_grad)
                        send_tensor_with_header(zero_grad, self.rank - 1, self.pp_group)

        # FIXED: Clear the lists to free memory
        input_tensors.clear()
        output_tensors.clear()
        torch.cuda.empty_cache()
        # Synchronize gradients across all stages
        if requires_grad_sync:
            dist.barrier(group=self.pp_group)
        
        # FIXED: Apply gradient clipping to prevent exploding gradients
        if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Calculate average accuracy and loss for the last stage
        if self.is_last_stage:
            acc /= len(batches)
            logging_loss /= len(batches)
        else:
            # Non-last stages should return 0 for consistency
            acc = 0.0
            logging_loss = 0.0

        # Return the average loss and accuracy
        return logging_loss, acc

    def train_one_forward_one_backward(self, data_loader, pgm, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        
        pipeline_depth = self.world_size 
        
        batches = list(data_loader)
        num_micro_batches = len(batches)
        
        fwd_inputs_q = deque()
        fwd_outputs_q = deque()

        total_loss = 0.0
        total_correct = 0
        
        self.print_layer_debug_norms(self.model, self.rank, f"Epoch {epoch+1} Start")

        # Main loop processes micro-batches
        for i in tqdm(range(num_micro_batches + pipeline_depth - 1)):
            
            # --- FORWARD PASS ---
            is_fwd_step = i < num_micro_batches
            if is_fwd_step:
                if self.is_first_stage:
                    input_tensor = batches[i]['image'].to(self.device)
                    # input_tensor.requires_grad = True # Not needed for first stage
                    output_tensor = self.model(input_tensor)
                    
                    if not self.is_last_stage:
                        send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                else:
                    input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group)
                    input_tensor.requires_grad = True
                    output_tensor = self.model(input_tensor)
                    if not self.is_last_stage:
                        send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)

                fwd_inputs_q.append(input_tensor)
                fwd_outputs_q.append(output_tensor)
            
            # --- BACKWARD PASS ---
            is_bwd_step = i >= pipeline_depth - 1
            if is_bwd_step:
                input_tensor_for_grad = fwd_inputs_q.popleft()
                output_tensor_for_grad = fwd_outputs_q.popleft()

                if self.is_last_stage:
                    batch_idx = i - pipeline_depth + 1
                    target = batches[batch_idx]['label'].to(self.device, dtype=torch.long)
                    loss = self.criterion(output_tensor_for_grad, target)
                    
                    total_loss += loss.item()
                    total_correct += (output_tensor_for_grad.argmax(dim=1) == target).sum().item()
                    
                    scaled_loss = loss / num_micro_batches
                    scaled_loss.backward()

                    # Send gradient to previous stage (if not first stage overall)
                    if not self.is_first_stage and input_tensor_for_grad is not None:
                        if input_tensor_for_grad.grad is not None:
                            send_tensor_with_header(input_tensor_for_grad.grad, self.rank - 1, self.pp_group)
                        else:
                            zero_grad = torch.zeros_like(input_tensor_for_grad)
                            send_tensor_with_header(zero_grad, self.rank - 1, self.pp_group)
                else:
                    grad_buffer = recv_tensor_with_header(self.rank + 1, self.device, self.pp_group)
                    output_tensor_for_grad.backward(gradient=grad_buffer)
                    
                    if not self.is_first_stage and input_tensor_for_grad is not None:
                        if input_tensor_for_grad.grad is not None:
                            send_tensor_with_header(input_tensor_for_grad.grad, self.rank - 1, self.pp_group)
                        else:
                            zero_grad = torch.zeros_like(input_tensor_for_grad)
                            send_tensor_with_header(zero_grad, self.rank - 1, self.pp_group)

        if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        
        self.print_layer_debug_norms(self.model, self.rank, f"Epoch {epoch+1} End")

        self.optimizer.zero_grad()
        
        if self.is_last_stage:
            avg_loss = total_loss / num_micro_batches
            
            total_samples = sum(len(b['label']) for b in batches)
            accuracy = (total_correct / total_samples) * 100
            
            return avg_loss, accuracy
        else:
            return 0.0, 0.0
            
    def print_layer_debug_norms(self, model, rank, point_in_time: str):
        """
        Prints the L2 norm of each layer's parameters and their gradients.

        Args:
            model: The PipelineParallelWrapper instance for the current rank.
            rank: The rank of the current process.
            point_in_time: A string describing when this is being called (e.g., "After backward").
        """
        print(f"--- [Rank {rank}] Layer Norms at '{point_in_time}' ---")
        
        # Access the local module within the wrapper
        params = list(model.local_module.named_parameters())
        
        if not params:
            print(f"  No parameters on this rank.")
            return

        for name, p in params:
            if p.requires_grad:
                param_norm = p.detach().data.norm(2).item()
                
                grad_norm_str = "N/A (None)"
                if p.grad is not None:
                    grad_norm = p.grad.detach().data.norm(2).item()
                    grad_norm_str = f"{grad_norm:.6f}"
                
                print(f"  Layer: {name:<40} | Param Norm: {param_norm:.6f} | Grad Norm: {grad_norm_str}")
            
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
    #             input_tensor = None # First stage has no input tensor from another stage
                
    #             # Store the model's direct input for the backward pass
    #             # This requires saving the original input `batch['image']`
    #             # and making it require grad if it's the very first stage
    #             # This part of the logic seems missing in the original code but is crucial.
    #             # For simplicity, we continue storing `None` as per the original code's structure.
                
    #         else:
    #             # THIS IS THE FIX: Receive from the PREVIOUS stage
    #             input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
    #             input_tensor.requires_grad = True # Make sure to set this for the backward pass
    #             output_tensor = self.model(input_tensor)
                
    #             if not self.is_last_stage:
    #                 # If not the last stage, send to the next one
    #                 send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)

    #         input_tensors.append(input_tensor)
    #         output_tensors.append(output_tensor)

    #     # All backward passes (in reverse order)
    #     for i in range(len(batches) - 1, -1, -1):
    #         batch = batches[i]
            
    #         # Pop the corresponding tensors for this micro-batch
    #         input_tensor_for_grad = input_tensors.pop()
    #         output_tensor = output_tensors.pop()
            
    #         if self.is_last_stage:
    #             # Get target labels based on batch structure
    #             target = batch.get("labels", batch.get("label")).to(device, dtype=torch.long)
                
    #             loss = self.criterion(output_tensor, target)
    #             logging_loss += loss.item()
    #             loss.backward()
                
    #             # Use the correct output tensor for accuracy calculation
    #             acc += (output_tensor.argmax(dim=1) == target).float().mean().item()
                
    #             if input_tensor_for_grad is not None:
    #                 grad_to_send = input_tensor_for_grad.grad
    #                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
    #         else:
    #             # Receive gradient from next stage
    #             # The dtype should match the output tensor's dtype
    #             grad_buffer = recv_tensor_with_header(self.rank + 1, device, self.pp_group, dtype=output_tensor.dtype)
    #             output_tensor.backward(gradient=grad_buffer)
                
    #             if not self.is_first_stage and input_tensor_for_grad is not None:
    #                 grad_to_send = input_tensor_for_grad.grad
    #                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)

    #     # Synchronize gradients across all stages
    #     if requires_grad_sync:
    #         dist.barrier(group=self.pp_group)
        
    #     # Optimizer step
    #     self.optimizer.step()
        
    #     # Calculate average accuracy for the last stage
    #     if self.is_last_stage:
    #         acc /= len(batches)

    #     # Return the average loss and accuracy
    #     return logging_loss / len(batches), acc if self.is_last_stage else 0.0



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
