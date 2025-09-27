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
import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict
import time

class PipelineDebugger:
    """Comprehensive debugging utility for pipeline parallelism"""
    
    def __init__(self, rank, pp_group, world_size):
        self.rank = rank
        self.pp_group = pp_group
        self.world_size = world_size
        self.is_first_stage = (rank == 0)
        self.is_last_stage = (rank == world_size - 1)
        
        # Debug counters and accumulators
        self.forward_count = 0
        self.backward_count = 0
        self.tensor_stats = defaultdict(list)
        
        def log_tensor_stats(self, tensor, name, step=""):
        """Log comprehensive tensor statistics"""
        if tensor is None:
            print(f"[DEBUG Rank {self.rank}] {step} {name}: NONE")
            return
            
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'norm': tensor.norm().item(),
            'shape': list(tensor.shape),
            'requires_grad': tensor.requires_grad,
            'has_grad': tensor.grad is not None if tensor.requires_grad else False
        }

                # Check for problematic values
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        print(f"[DEBUG Rank {self.rank}] {step} {name}:")
        print(f"  Shape: {stats['shape']}, Requires_grad: {stats['requires_grad']}")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
        print(f"  Norm: {stats['norm']:.6f}")
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
        
        if tensor.requires_grad and tensor.grad is not None:
            grad_norm = tensor.grad.norm().item()
            grad_mean = tensor.grad.mean().item()
            print(f"  Grad Norm: {grad_norm:.6f}, Grad Mean: {grad_mean:.6f}")
        
        return stats
    
    def check_gradient_flow(self, model, step=""):
        """Check gradient flow through the model"""
        print(f"\n[GRAD DEBUG Rank {self.rank}] {step} Gradient Flow Check:")
        
        total_grad_norm = 0.0
        layer_count = 0
        zero_grad_layers = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_count += 1
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    
                    # Check for problematic gradients
                    has_nan_grad = torch.isnan(param.grad).any().item()
                    has_inf_grad = torch.isinf(param.grad).any().item()
                    
                    if grad_norm < 1e-8:
                        zero_grad_layers += 1
                    
                    print(f"  {name}: grad_norm={grad_norm:.8f}, "
                          f"NaN={has_nan_grad}, Inf={has_inf_grad}")
                else:
                    zero_grad_layers += 1
                    print(f"  {name}: NO GRADIENT")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total layers: {layer_count}, Zero grad layers: {zero_grad_layers}")
        print(f"  Total gradient norm: {total_grad_norm:.8f}")
        
        return total_grad_norm, zero_grad_layers, layer_count
    
    def verify_data_sync(self, data, name, step=""):
        """Verify data synchronization across ranks"""
        if data is None:
            print(f"[SYNC DEBUG] {step} {name}: Data is None on rank {self.rank}")
            return
            
        # Compute local checksum
        local_sum = data.sum().item()
        local_mean = data.mean().item()
        
        # Gather from all ranks
        gathered_sums = [None] * self.world_size
        gathered_means = [None] * self.world_size
        
        dist.all_gather_object(gathered_sums, local_sum, group=self.pp_group)
        dist.all_gather_object(gathered_means, local_mean, group=self.pp_group)
        
        if self.rank == 0:
            print(f"[SYNC DEBUG] {step} {name} - Sums across ranks: {gathered_sums}")
            print(f"[SYNC DEBUG] {step} {name} - Means across ranks: {gathered_means}")
            
            # Check consistency
            sum_diff = max(gathered_sums) - min(gathered_sums)
            mean_diff = max(gathered_means) - min(gathered_means)
            
            if sum_diff < 1e-6:
                print(f"[SYNC DEBUG] ✓ {name} is synchronized across ranks")
            else:
                print(f"[SYNC DEBUG] ✗ {name} differs across ranks! Sum diff: {sum_diff}")

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
from collections import deque

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
    
    from collections import deque

    def train_one_forward_one_backward(self, data_loader, pgm):
        self.model.train()
        self.optimizer.zero_grad()
        
        # The number of micro-batches to process before starting backward passes
        # This is also the number of pipeline stages.
        pipeline_depth = self.world_size 
        
        batches = list(data_loader)
        num_micro_batches = len(batches)
        
        # Queues to hold data for forward and backward passes for each stage
        fwd_inputs_q = deque()
        fwd_outputs_q = deque()

        # --- NEW: Accumulators for metrics ---
        total_loss = 0.0
        total_correct = 0

        # Main loop processes micro-batches
        for i in range(num_micro_batches + pipeline_depth - 1):
            # --- FORWARD PASS ---
            is_fwd_step = i < num_micro_batches
            if is_fwd_step:
                if self.is_first_stage:
                    # First stage gets data from loader
                    input_tensor = batches[i]['image'].to(self.device)
                    output_tensor = self.model(input_tensor)
                    send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                else:
                    input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group)
                    input_tensor.requires_grad = True
                    output_tensor = self.model(input_tensor)
                    if not self.is_last_stage:
                        send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)

                # Save tensors needed for the corresponding backward pass
                fwd_inputs_q.append(input_tensor)
                fwd_outputs_q.append(output_tensor)
            
            # --- BACKWARD PASS ---
            # Start backward passes after the pipeline is full
            is_bwd_step = i >= pipeline_depth - 1
            if is_bwd_step:
                # Get the tensors from the corresponding forward pass
                input_tensor_for_grad = fwd_inputs_q.popleft()
                output_tensor_for_grad = fwd_outputs_q.popleft()

                if self.is_last_stage:
                    target = batches[i - pipeline_depth + 1]['label'].to(self.device, dtype=torch.long)
                    loss = self.criterion(output_tensor_for_grad, target)
                    
                    # --- NEW: Accumulate loss and accuracy ---
                    total_loss += loss.item()
                    total_correct += (output_tensor_for_grad.argmax(dim=1) == target).sum().item()
                    
                    # We scale the loss by the number of micro-batches to average gradients
                    # This is important if your loss function is 'sum' instead of 'mean'
                    scaled_loss = loss / num_micro_batches
                    scaled_loss.backward()

                    if input_tensor_for_grad is not None:
                        # Scale the gradient before sending
                        grad_to_send = input_tensor_for_grad.grad / num_micro_batches
                        send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
                else:
                    grad_buffer = recv_tensor_with_header(self.rank + 1, self.device, self.pp_group)
                    output_tensor_for_grad.backward(gradient=grad_buffer)
                    
                    if not self.is_first_stage and input_tensor_for_grad is not None:
                        # Note: grad is already scaled from the previous stage
                        send_tensor_with_header(input_tensor_for_grad.grad, self.rank - 1, self.pp_group)

        # After the loop, step the optimizer
        self.optimizer.step()

        # --- NEW: Calculate and return final metrics ---
        if self.is_last_stage:
            # Calculate average loss over all micro-batches
            avg_loss = total_loss / num_micro_batches
            
            # Calculate accuracy over the entire dataset
            total_samples = sum(len(b['label']) for b in batches)
            accuracy = (total_correct / total_samples) * 100
            
            return avg_loss, accuracy
        else:
            # Other stages don't have the final metrics
            return 0.0, 0.0
            
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
        
        
