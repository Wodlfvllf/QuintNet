# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from .Processgroup import ProcessGroupManager
# from .pp_wrapper import PipelineParallelWrapper
# from .operations import Send, Recv
# import torch
# import torch.distributed as dist
# from tqdm import tqdm

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from collections import deque
# from tqdm import tqdm
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from collections import deque
# from tqdm import tqdm
# from typing import Optional, Tuple

# def send_tensor_with_header(tensor: torch.Tensor, dst: int, group=None):
#     """Sends a tensor preceded by a header containing its shape and dtype info."""
#     if tensor is None:
#         # Send a flag indicating None
#         flag = torch.tensor([-1], dtype=torch.long, device='cuda')
#         dist.send(tensor=flag, dst=dst, group=group)
#         return
    
#     # Send flag indicating valid tensor
#     flag = torch.tensor([1], dtype=torch.long, device=tensor.device)
#     dist.send(tensor=flag, dst=dst, group=group)
    
#     # Send the number of dimensions
#     num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
#     dist.send(tensor=num_dims, dst=dst, group=group)

#     # Send the shape
#     shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
#     dist.send(tensor=shape_tensor, dst=dst, group=group)

#     # Send the tensor data
#     dist.send(tensor=tensor.contiguous(), dst=dst, group=group)

# def recv_tensor_with_header(src: int, device: torch.device, group=None, dtype=torch.float32) -> Optional[torch.Tensor]:
#     """Receives a tensor that is preceded by a shape and dtype header."""
#     # Receive flag
#     flag = torch.zeros(1, dtype=torch.long, device=device)
#     dist.recv(tensor=flag, src=src, group=group)
    
#     if flag.item() == -1:
#         return None
    
#     # Receive the number of dimensions
#     num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
#     dist.recv(tensor=num_dims_tensor, src=src, group=group)
#     num_dims = num_dims_tensor.item()

#     # Receive the shape
#     shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
#     dist.recv(tensor=shape_tensor, src=src, group=group)
#     tensor_shape = shape_tensor.tolist()
    
#     # Create a correctly shaped buffer and receive the tensor data
#     buffer = torch.zeros(tensor_shape, dtype=dtype, device=device)
#     dist.recv(tensor=buffer, src=src, group=group)
    
#     return buffer

# class PipelineTrainer:
#     def __init__(self, model, pp_group, criterion, device, optimizer=None, max_grad_norm=1.0):
#         """
#         Initializes the trainer for pipeline parallelism.
        
#         Args:
#             model: The model, already wrapped in PipelineParallelWrapper.
#             pp_group: The process group for the pipeline.
#             criterion: The loss function.
#             device: The device for the current rank.
#             optimizer: Optional pre-configured optimizer for the local model parameters.
#             max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
#         """
#         self.model = model
#         self.pp_group = pp_group
#         self.criterion = criterion
#         self.device = device
#         self.max_grad_norm = 0
        
#         # Initialize optimizer if not provided
#         if optimizer is None:
#             self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
#         else:
#             self.optimizer = optimizer
        
#         # Get rank and size from the specific pipeline group
#         self.rank = dist.get_rank(group=pp_group)
#         self.world_size = dist.get_world_size(group=pp_group)
        
#         # Flags for stage position
#         self.is_first_stage = (self.rank == 0)
#         self.is_last_stage = (self.rank == self.world_size - 1)
        
#         print(f"PipelineTrainer initialized - Rank: {self.rank}, Stage: {self.rank}/{self.world_size-1}")
#         print(f"  First stage: {self.is_first_stage}, Last stage: {self.is_last_stage}")
#         print(f"  Device: {device}, Max grad norm: {max_grad_norm}")

#     def train_step_naive(self, input_data, target):
#         """
#         Simple training step - one forward, one backward, one update.
#         Good for debugging but not efficient.
#         """
#         self.model.train()
#         self.optimizer.zero_grad()

#         input_tensor_for_stage = None
#         output_tensor_of_stage = None

#         # --- FORWARD PASS ---
#         if self.is_first_stage:
#             input_data = input_data.to(self.device)
#             output_tensor_of_stage = self.model(input_data)
            
#             if not self.is_last_stage:
#                 send_tensor_with_header(output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)
#         else:
#             # Receive from previous stage
#             input_tensor_for_stage = recv_tensor_with_header(
#                 self.rank - 1, self.device, self.pp_group, dtype=input_data.dtype
#             )
#             input_tensor_for_stage.requires_grad = True
            
#             output_tensor_of_stage = self.model(input_tensor_for_stage)
            
#             if not self.is_last_stage:
#                 send_tensor_with_header(output_tensor_of_stage.detach(), self.rank + 1, self.pp_group)

#         # --- BACKWARD PASS ---
#         if self.is_last_stage:
#             target = target.to(device=self.device, dtype=torch.long)
#             loss = self.criterion(output_tensor_of_stage, target)
#             loss.backward()
            
#             if not self.is_first_stage and input_tensor_for_stage is not None:
#                 grad_to_send = input_tensor_for_stage.grad
#                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
#         else:
#             # Receive gradient from next stage
#             grad_buffer = recv_tensor_with_header(
#                 self.rank + 1, self.device, self.pp_group, dtype=output_tensor_of_stage.dtype
#             )
            
#             output_tensor_of_stage.backward(gradient=grad_buffer)
            
#             if not self.is_first_stage and input_tensor_for_stage is not None:
#                 grad_to_send = input_tensor_for_stage.grad
#                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
        
#         # --- OPTIMIZER STEP ---
#         if self.max_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
#         self.optimizer.step()
        
#         # Return loss and accuracy for last stage
#         if self.is_last_stage:
#             with torch.no_grad():
#                 acc = (output_tensor_of_stage.argmax(dim=1) == target).float().mean().item()
#             return loss.item(), acc * 100
#         else:
#             return None, None
    
#     def train_all_forward_then_backward(self, data_loader, epoch, requires_grad_sync=True):
#         """
#         Optimized training: All forwards, then all backwards.
#         Better GPU utilization but higher memory usage.
#         """
#         self.model.train()
        
#         input_tensors = []
#         output_tensors = []
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0
        
#         # Convert data_loader to list
#         batches = list(data_loader)
#         num_batches = len(batches)
        
#         # Zero gradients once at the beginning
#         self.optimizer.zero_grad()
        
#         # === ALL FORWARD PASSES ===
#         print(f"[Rank {self.rank}] Starting {num_batches} forward passes...")
        
#         for i, batch in enumerate(tqdm(batches, desc=f"Forward (Rank {self.rank})", disable=(self.rank != 0))):
#             if self.is_first_stage:
#                 # Process input data
#                 if 'image' in batch:
#                     input_data = batch['image'].to(self.device)
#                 elif 'images' in batch:
#                     input_data = batch['images'].to(self.device)
#                 else:
#                     raise KeyError("No 'image' or 'images' key in batch")
                
#                 # No need to set requires_grad for first stage input
#                 output_tensor = self.model(input_data)
                
#                 if not self.is_last_stage:
#                     send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                
#                 # Store for backward pass (store None for first stage)
#                 input_tensor = None
#             else:
#                 # Receive from previous stage
#                 input_tensor = recv_tensor_with_header(
#                     self.rank - 1, self.device, self.pp_group, dtype=torch.float32
#                 )
#                 input_tensor.requires_grad = True
#                 output_tensor = self.model(input_tensor)
                
#                 if not self.is_last_stage:
#                     send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
            
#             # Store tensors for backward pass
#             input_tensors.append(input_tensor)
#             output_tensors.append(output_tensor)
        
#         # === ALL BACKWARD PASSES (in reverse order) ===
#         print(f"[Rank {self.rank}] Starting {num_batches} backward passes...")
        
#         for i in range(num_batches - 1, -1, -1):
#             batch = batches[i]
            
#             # Get tensors in LIFO order
#             input_tensor = input_tensors.pop()
#             output_tensor = output_tensors.pop()
            
#             if self.is_last_stage:
#                 # Get target labels
#                 if 'labels' in batch:
#                     target = batch['labels'].to(self.device, dtype=torch.long)
#                 elif 'label' in batch:
#                     target = batch['label'].to(self.device, dtype=torch.long)
#                 else:
#                     raise KeyError("No 'labels' or 'label' key in batch")
                
#                 # Calculate loss (unscaled for logging)
#                 loss = self.criterion(output_tensor, target)
#                 total_loss += loss.item()
                
#                 # Scale loss for gradient accumulation
#                 scaled_loss = loss / num_batches
#                 scaled_loss.backward()
                
#                 # Calculate accuracy
#                 with torch.no_grad():
#                     total_correct += (output_tensor.argmax(dim=1) == target).sum().item()
#                     total_samples += target.size(0)
                
#                 # Send gradient to previous stage
#                 if not self.is_first_stage and input_tensor is not None:
#                     if input_tensor.grad is not None:
#                         send_tensor_with_header(input_tensor.grad, self.rank - 1, self.pp_group)
#             else:
#                 # Receive gradient from next stage
#                 grad_buffer = recv_tensor_with_header(
#                     self.rank + 1, self.device, self.pp_group, dtype=output_tensor.dtype
#                 )
                
#                 # Scale gradient for accumulation
#                 scaled_grad = grad_buffer
#                 output_tensor.backward(gradient=scaled_grad)
                
#                 # Send gradient to previous stage
#                 if not self.is_first_stage and input_tensor is not None:
#                     if input_tensor.grad is not None:
#                         send_tensor_with_header(input_tensor.grad, self.rank - 1, self.pp_group)
        
#         # Clear lists to free memory
#         input_tensors.clear()
#         output_tensors.clear()
        
#         # Optional memory cleanup
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         # Synchronize before optimizer step
#         if requires_grad_sync:
#             dist.barrier(group=self.pp_group)
        
#         # Gradient clipping
#         if self.max_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
#         # Optimizer step
#         self.optimizer.step()
        
#         # Calculate metrics
#         if self.is_last_stage:
#             avg_loss = total_loss / num_batches
#             accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
#             return avg_loss, accuracy
#         else:
#             return 0.0, 0.0

#     # This is your new, refactored function
#     def train_1f1b_three_phase(self, data_loader, epoch):
#         """
#         Implements the 1F1B schedule using an explicit three-phase structure:
#         1. Warmup: Fills the pipeline with forward passes.
#         2. Steady State: Interleaves one forward and one backward pass per step.
#         3. Cooldown: Drains the pipeline with remaining backward passes.
#         """
#         self.model.train()
#         self.optimizer.zero_grad()

#         batches = list(data_loader)
#         num_micro_batches = len(batches)

#         # --- 1. CALCULATE PHASE BOUNDARIES ---
#         # This logic determines how many steps are in each phase for each rank
#         num_warmup_microbatches = self.world_size - self.rank - 1
#         # Ensure warmup steps don't exceed the total number of batches
#         num_warmup_microbatches = min(num_warmup_microbatches, num_micro_batches)
        
#         num_cooldown_microbatches = self.world_size - self.rank - 1
#         num_cooldown_microbatches = min(num_cooldown_microbatches, num_micro_batches)

#         num_steady_microbatches = num_micro_batches - num_warmup_microbatches
        
#         # Queues for storing tensors for the backward pass
#         fwd_inputs_q = deque()
#         fwd_outputs_q = deque()

#         # Accumulators for metrics (only used on the last stage)
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0
        
#         # Use tqdm only on the first rank to avoid messy prints
#         if self.rank == 0:
#             pbar = tqdm(total=num_micro_batches, desc=f"Epoch {epoch} PP-1F1B")

#         # ===================================================================
#         # Helper functions to encapsulate forward and backward step logic
#         # ===================================================================
        
#         def _forward_step(micro_batch_idx):
#             """Performs a single forward pass for a given micro-batch."""
#             batch = batches[micro_batch_idx]
            
#             if self.is_first_stage:
#                 if 'image' in batch:
#                     input_tensor = batch['image'].to(self.device)
#                 else: # Assuming 'images' as a fallback
#                     input_tensor = batch['images'].to(self.device)
                
#                 output_tensor = self.model(input_tensor)
                
#                 if not self.is_last_stage:
#                     send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                
#                 # The first stage's input doesn't come from another stage
#                 return None, output_tensor
#             else:
#                 input_tensor = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group)
#                 input_tensor.requires_grad = True
                
#                 output_tensor = self.model(input_tensor)
                
#                 if not self.is_last_stage:
#                     send_tensor_with_header(output_tensor.detach(), self.rank + 1, self.pp_group)
                
#                 return input_tensor, output_tensor

#         def _backward_step(micro_batch_idx):
#             """Performs a single backward pass for a given micro-batch."""
#             nonlocal total_loss, total_correct, total_samples

#             input_tensor_for_grad = fwd_inputs_q.popleft()
#             output_tensor_for_grad = fwd_outputs_q.popleft()

#             if self.is_last_stage:
#                 batch = batches[micro_batch_idx]
#                 if 'labels' in batch:
#                     target = batch['labels'].to(self.device, dtype=torch.long)
#                 else: # Assuming 'label' as a fallback
#                     target = batch['label'].to(self.device, dtype=torch.long)

#                 loss = self.criterion(output_tensor_for_grad, target)
#                 total_loss += loss.item()
                
#                 with torch.no_grad():
#                     total_correct += (output_tensor_for_grad.argmax(dim=1) == target).sum().item()
#                     total_samples += target.size(0)

#                 scaled_loss = loss / num_micro_batches
#                 scaled_loss.backward()
                
#                 grad_to_send = input_tensor_for_grad.grad
#                 send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
#             else:
#                 grad_buffer = recv_tensor_with_header(self.rank + 1, self.device, self.pp_group)
#                 output_tensor_for_grad.backward(gradient=grad_buffer)
                
#                 if not self.is_first_stage:
#                     grad_to_send = input_tensor_for_grad.grad
#                     send_tensor_with_header(grad_to_send, self.rank - 1, self.pp_group)
        
#         # ===================================================================
#         # The Three Phases
#         # ===================================================================
#         print(f"[Rank {self.rank}] Starting training with 1F1B three-phase schedule...")
#         # --- PHASE 1: WARMUP ---
#         # Only forward passes to fill the pipeline.
#         for i in range(num_warmup_microbatches):
#             print(f"[Rank {self.rank}] Warmup forward pass {i+1}/{num_warmup_microbatches}")
#             input_tensor, output_tensor = _forward_step(micro_batch_idx=i)
#             fwd_inputs_q.append(input_tensor)
#             fwd_outputs_q.append(output_tensor)
#             print(f"[Rank {self.rank}] Warmup forward pass {i+1}/{num_warmup_microbatches} completed")
#             # After warmup phase
#             if self.rank == 0:
#                 pbar.update(num_warmup_microbatches)

#         # --- PHASE 2: STEADY STATE ---
#         # One forward pass and one backward pass per step.
#         for i in range(num_steady_microbatches):
#             print(f"[Rank {self.rank}] Steady state step {i+1}/{num_steady_microbatches}")
#             fwd_batch_idx = i + num_warmup_microbatches
            
#             input_tensor, output_tensor = _forward_step(micro_batch_idx=fwd_batch_idx)
#             fwd_inputs_q.append(input_tensor)
#             fwd_outputs_q.append(output_tensor)
#             print(f"[Rank {self.rank}] Steady state forward pass {i+1}/{num_steady_microbatches} completed")
#             bwd_batch_idx = i
#             print(f"[Rank {self.rank}] Steady state backward pass {i+1}/{num_steady_microbatches} starting")
#             _backward_step(micro_batch_idx=bwd_batch_idx)
#             print(f"[Rank {self.rank}] Steady state backward pass {i+1}/{num_steady_microbatches} completed")

#             if self.rank == 0:
#                 pbar.update(1)

#         # --- PHASE 3: COOLDOWN ---
#         # Only backward passes to drain the pipeline.
#         for i in range(num_cooldown_microbatches):
#             print(f"[Rank {self.rank}] Cooldown backward pass {i+1}/{num_cooldown_microbatches} starting")
#             bwd_batch_idx = i + num_steady_microbatches
#             _backward_step(micro_batch_idx=bwd_batch_idx)
#             print(f"[Rank {self.rank}] Cooldown backward pass {i+1}/{num_cooldown_microbatches} completed")

#             if self.rank == 0:
#                 pbar.update(1)

#         if self.rank == 0:
#             pbar.close()

#         # ===================================================================
#         # Finalization
#         # ===================================================================
#         if self.max_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
#         print(f"[Rank {self.rank}] Completed all forward and backward passes.")
#         self.optimizer.step()
#         print(f"[Rank {self.rank}] Optimizer step completed.")
        
#         if self.is_last_stage:
#             avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0
#             accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
#             return avg_loss, accuracy
#         else:
#             return 0.0, 0.0
            
#     def print_layer_debug_norms(self, model, rank, point_in_time: str):
#         """
#         Prints the L2 norm of each layer's parameters and their gradients.

#         Args:
#             model: The PipelineParallelWrapper instance for the current rank.
#             rank: The rank of the current process.
#             point_in_time: A string describing when this is being called (e.g., "After backward").
#         """
#         print(f"--- [Rank {rank}] Layer Norms at '{point_in_time}' ---")
        
#         # Access the local module within the wrapper
#         params = list(model.local_module.named_parameters())
        
#         if not params:
#             print(f"  No parameters on this rank.")
#             return

#         for name, p in params:
#             if p.requires_grad:
#                 param_norm = p.detach().data.norm(2).item()
                
#                 grad_norm_str = "N/A (None)"
#                 if p.grad is not None:
#                     grad_norm = p.grad.detach().data.norm(2).item()
#                     grad_norm_str = f"{grad_norm:.6f}"
                
#                 print(f"  Layer: {name:<40} | Param Norm: {param_norm:.6f} | Grad Norm: {grad_norm_str}")
            
#     def evaluate(self, val_loader):
#         self.model.eval()
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 images = batch['image']
#                 targets = batch['label']

#                 if self.is_first_stage:
#                     output = self.model(images.to(self.device))
#                     if not self.is_last_stage:
#                         # MODIFIED: Use the helper for evaluation as well.
#                         send_tensor_with_header(output, self.rank + 1, self.pp_group)
#                 else:
#                     # MODIFIED: The helper removes the need for get_input_shape.
#                     buffer = recv_tensor_with_header(self.rank - 1, self.device, self.pp_group, dtype=images.dtype)
#                     output = self.model(buffer)
#                     if not self.is_last_stage:
#                         send_tensor_with_header(output, self.rank + 1, self.pp_group)

#                 if self.is_last_stage:
#                     targets = targets.to(self.device, dtype=torch.long)
#                     loss = self.criterion(output, targets)
#                     total_loss += loss.item() * output.size(0)
#                     total_correct += (output.argmax(dim=1) == targets).sum().item()
#                     total_samples += output.size(0)

#         dist.barrier(group=self.pp_group)
        
#         if self.is_last_stage:
#             avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
#             accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
#             return avg_loss, accuracy
#         else:
#             return None, None

import torch
import torch.distributed as dist
import torch.nn as nn
from collections import deque
from tqdm import tqdm
from typing import Optional, Tuple

# ============================================================================
# NON-BLOCKING COMMUNICATION FUNCTIONS
# ============================================================================

def pipeline_send(tensor: torch.Tensor, dst: int, group=None):
    """Non-blocking send operation."""
    if tensor is None:
        flag = torch.tensor([-1], dtype=torch.long, device='cuda')
        op = dist.P2POp(dist.isend, flag, dst, group=group)
        reqs = dist.batch_isend_irecv([op])
        for req in reqs:
            req.wait()
        return
    
    flag = torch.tensor([1], dtype=torch.long, device=tensor.device)
    num_dims = torch.tensor([tensor.dim()], dtype=torch.long, device=tensor.device)
    shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    
    ops = [
        dist.P2POp(dist.isend, flag, dst, group=group),
        dist.P2POp(dist.isend, num_dims, dst, group=group),
        dist.P2POp(dist.isend, shape_tensor, dst, group=group),
        dist.P2POp(dist.isend, tensor.contiguous(), dst, group=group)
    ]
    
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()


def pipeline_recv(src: int, device: torch.device, group=None, dtype=torch.float32) -> Optional[torch.Tensor]:
    """Non-blocking receive operation."""
    flag = torch.zeros(1, dtype=torch.long, device=device)
    op = dist.P2POp(dist.irecv, flag, src, group=group)
    reqs = dist.batch_isend_irecv([op])
    for req in reqs:
        req.wait()
    
    if flag.item() == -1:
        return None
    
    num_dims_tensor = torch.zeros(1, dtype=torch.long, device=device)
    op = dist.P2POp(dist.irecv, num_dims_tensor, src, group=group)
    reqs = dist.batch_isend_irecv([op])
    for req in reqs:
        req.wait()
    num_dims = num_dims_tensor.item()
    
    shape_tensor = torch.zeros(num_dims, dtype=torch.long, device=device)
    op = dist.P2POp(dist.irecv, shape_tensor, src, group=group)
    reqs = dist.batch_isend_irecv([op])
    for req in reqs:
        req.wait()
    tensor_shape = shape_tensor.tolist()
    
    buffer = torch.zeros(tensor_shape, dtype=dtype, device=device)
    op = dist.P2POp(dist.irecv, buffer, src, group=group)
    reqs = dist.batch_isend_irecv([op])
    for req in reqs:
        req.wait()
    
    torch.cuda.synchronize()
    return buffer


# ============================================================================
# PIPELINE TRAINER CLASS
# ============================================================================

class PipelineTrainer:
    def __init__(self, model, pp_group, criterion, device, optimizer=None, max_grad_norm=1.0):
        """
        Initializes the trainer for pipeline parallelism with non-blocking communication.
        
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
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.rank = dist.get_rank(group=pp_group)
        self.world_size = dist.get_world_size(group=pp_group)
        
        self.is_first_stage = (self.rank == 0)
        self.is_last_stage = (self.rank == self.world_size - 1)
        
        print(f"PipelineTrainer initialized - Rank: {self.rank}, Stage: {self.rank}/{self.world_size-1}")
        print(f"  First stage: {self.is_first_stage}, Last stage: {self.is_last_stage}")
        print(f"  Device: {device}, Max grad norm: {max_grad_norm}")


    # def train_1f1b_three_phase(self, data_loader, epoch):
    #     """
    #     Implements the 1F1B schedule with non-blocking communication.
        
    #     Three phases:
    #     1. Warmup: Fill the pipeline with forward passes
    #     2. Steady State: Interleave forward and backward passes (1F1B)
    #     3. Cooldown: Drain the pipeline with remaining backward passes
    #     """
    #     self.model.train()
    #     self.optimizer.zero_grad()

    #     batches = list(data_loader)
    #     # --- inside train_1f1b_three_phase, after batches = list(data_loader) ---
    #     num_micro_batches = len(batches)
    #     # GLOBAL steady count (same for all ranks). This is the correct steady-phase size:
    #     num_steady_microbatches = max(0, num_micro_batches - (self.world_size - 1))
    #     # Per-rank warmup/cooldown (still rank-dependent)
    #     num_warmup_microbatches = min(self.world_size - self.rank - 1, num_micro_batches)
    #     # cooldown: remaining microbatches for this rank (total - warmup - steady)
    #     num_cooldown_microbatches = max(0, num_micro_batches - num_warmup_microbatches - num_steady_microbatches)

    #     # Logging for sanity
    #     print(f"[Rank {self.rank}] warmup={num_warmup_microbatches}, steady={num_steady_microbatches}, cooldown={num_cooldown_microbatches}, total={num_micro_batches}")

        
    #     # Queues for storing tensors
    #     fwd_inputs_q = deque()
    #     fwd_outputs_q = deque()

    #     # Metrics accumulators
    #     total_loss = 0.0
    #     total_correct = 0
    #     total_samples = 0
        
    #     if self.rank == 0:
    #         pbar = tqdm(total=num_micro_batches, desc=f"Epoch {epoch} PP-1F1B")

    #     # ===================================================================
    #     # PHASE 1: WARMUP - Fill the pipeline
    #     # ===================================================================
    #     print(f"[Rank {self.rank}] Starting warmup phase with {num_warmup_microbatches} micro-batches...")
    #     for i in range(num_warmup_microbatches):
    #         batch = batches[i]
    #         print(f"[Rank {self.rank}] Warmup forward pass {i+1}/{num_warmup_microbatches}")
    #         if self.is_first_stage:
    #             print(f"[Rank {self.rank}] Processing input data for warmup")
    #             # First stage: process input data
    #             input_tensor = batch['image'].to(self.device) if 'image' in batch else batch['images'].to(self.device)
    #             print(f"[Rank {self.rank}] Input tensor shape: {input_tensor.shape}")
    #             output_tensor = self.model(input_tensor)
    #             print(f"[Rank {self.rank}] Output tensor shape: {output_tensor.shape}")
                
    #             print(f"[Rank {self.rank}] Completed model forward for warmup")
    #             if not self.is_last_stage:
    #                 print(f"[Rank {self.rank}] Sending output to next stage")
    #                 pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
    #                 print(f"[Rank {self.rank}] Sent output to next stage")
                
    #             fwd_inputs_q.append(None)
    #             fwd_outputs_q.append(output_tensor)
    #         else:
    #             # Non-first stages: receive from previous stage
    #             print(f"[Rank {self.rank}] Receiving input from previous stage for warmup")
    #             input_tensor = pipeline_recv(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
    #             print(f"[Rank {self.rank}] Received input tensor shape: {input_tensor.shape}")
    #             input_tensor.requires_grad = True
    #             print(f"[Rank {self.rank}] Received input, processing model forward for warmup")
    #             output_tensor = self.model(input_tensor)
    #             print(f"[Rank {self.rank}] Output tensor shape: {output_tensor.shape}")
    #             print(f"[Rank {self.rank}] Completed model forward for warmup")
    #             if not self.is_last_stage:
    #                 print(f"[Rank {self.rank}] Sending output to next stage")
    #                 pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
    #                 print(f"[Rank {self.rank}] Sent output to next stage")
                
    #             fwd_inputs_q.append(input_tensor)
    #             fwd_outputs_q.append(output_tensor)
        
    #     # Update progress bar after warmup
    #     if self.rank == 0 and num_warmup_microbatches > 0:
    #         pbar.update(num_warmup_microbatches)

    #     # ===================================================================
    #     # PHASE 2: STEADY STATE - 1F1B interleaving
    #     # ===================================================================
    #     print(f"[Rank {self.rank}] Starting steady state phase with {num_steady_microbatches} micro-batches...")
    #     for i in range(num_steady_microbatches):
    #         fwd_batch_idx = i + num_warmup_microbatches
    #         bwd_batch_idx = i
    #         print(f"[Rank {self.rank}] Steady state step {i+1}/{num_steady_microbatches}")
    #         # ---------------------------------------------------------------
    #         # FORWARD STEP
    #         # ---------------------------------------------------------------
    #         batch_fwd = batches[fwd_batch_idx]
            
    #         if self.is_first_stage:
    #             # First stage: process new input
    #             print(f"[Rank {self.rank}] Processing input data for steady state forward")
    #             input_tensor = batch_fwd['image'].to(self.device) if 'image' in batch_fwd else batch_fwd['images'].to(self.device)
    #             print(f"[Rank {self.rank}] Input tensor shape: {input_tensor.shape}")
    #             output_tensor = self.model(input_tensor)
    #             print(f"[Rank {self.rank}] Output tensor shape: {output_tensor.shape}")
                
    #             if not self.is_last_stage:
    #                 print(f"[Rank {self.rank}] Sending output to next stage")
    #                 pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
    #                 print(f"[Rank {self.rank}] Sent output to next stage")
                
    #             fwd_inputs_q.append(None)
    #             fwd_outputs_q.append(output_tensor)
                
    #         else:
    #             # Non-first stages: receive from previous stage
    #             print(f"[Rank {self.rank}] Receiving input from previous stage for steady state forward")
    #             input_tensor = pipeline_recv(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
    #             input_tensor.requires_grad = True
    #             print(f"[Rank {self.rank}] Received input tensor shape: {input_tensor.shape}")
    #             print(f"[Rank {self.rank}] Received input, processing model forward for steady state")
                
    #             output_tensor = self.model(input_tensor)
    #             print(f"[Rank {self.rank}] Output tensor shape: {output_tensor.shape}")
    #             print(f"[Rank {self.rank}] Completed model forward for steady state")
    #             if not self.is_last_stage:
    #                 print(f"[Rank {self.rank}] Sending output to next stage")
    #                 pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
    #                 print(f"[Rank {self.rank}] Sent output to next stage")
                
    #             fwd_inputs_q.append(input_tensor)
    #             fwd_outputs_q.append(output_tensor)
            
    #         # ---------------------------------------------------------------
    #         # BACKWARD STEP
    #         # ---------------------------------------------------------------
    #         print(f"[Rank {self.rank}] Starting backward step {i+1}/{num_steady_microbatches}")
    #         batch_bwd = batches[bwd_batch_idx]
    #         input_tensor_bwd = fwd_inputs_q.popleft()
    #         output_tensor_bwd = fwd_outputs_q.popleft()
            
    #         if self.is_last_stage:
    #             # Last stage: compute loss and send gradient
    #             print(f"[Rank {self.rank}] Computing loss for steady state backward")
    #             target = batch_bwd['label'].to(self.device, dtype=torch.long) if 'label' in batch_bwd else batch_bwd['labels'].to(self.device, dtype=torch.long)
                
    #             loss = self.criterion(output_tensor_bwd, target)
    #             total_loss += loss.item()
    #             print(f"[Rank {self.rank}] Loss computed: {loss.item()}")
    #             with torch.no_grad():
    #                 total_correct += (output_tensor_bwd.argmax(dim=1) == target).sum().item()
    #                 total_samples += target.size(0)
                
    #             # Scale loss for gradient accumulation
    #             scaled_loss = loss / num_micro_batches
    #             scaled_loss.backward()
                
    #             # Send gradient to previous stage
    #             if not self.is_first_stage and input_tensor_bwd is not None:
    #                 if input_tensor_bwd.grad is not None:
    #                     print(f"[Rank {self.rank}] Sending gradient to previous stage for steady state backward")
    #                     pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
    #                     print(f"[Rank {self.rank}] Sent gradient to previous stage for steady state backward")
                    
    #         else:
    #             print(f"[Rank {self.rank}] Receiving gradient from next stage for steady state backward")
    #             # Non-last stages: receive gradient and backprop
    #             grad_buffer = pipeline_recv(self.rank + 1, self.device, self.pp_group, dtype=output_tensor_bwd.dtype)
    #             print(f"[Rank {self.rank}] Received gradient tensor shape: {grad_buffer.shape}")
    #             output_tensor_bwd.backward(gradient=grad_buffer)
    #             print(f"[Rank {self.rank}] Backward pass completed for steady state")
    #             # Send gradient to previous stage (if not first stage)
    #             if not self.is_first_stage and input_tensor_bwd is not None:
    #                 if input_tensor_bwd.grad is not None:
    #                     pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
            
    #         if self.rank == 0:
    #             pbar.update(1)

    #     # ===================================================================
    #     # PHASE 3: COOLDOWN - Drain the pipeline
    #     # ===================================================================
    #     print(f"[Rank {self.rank}] Starting cooldown phase with {num_cooldown_microbatches} micro-batches...")  
    #     for i in range(num_cooldown_microbatches):
    #         print(f"[Rank {self.rank}] Cooldown backward pass {i+1}/{num_cooldown_microbatches} starting")
    #         bwd_batch_idx = i + num_steady_microbatches
    #         batch_bwd = batches[bwd_batch_idx]
            
    #         input_tensor_bwd = fwd_inputs_q.popleft()
    #         output_tensor_bwd = fwd_outputs_q.popleft()
            
    #         if self.is_last_stage:
    #             print(f"[Rank {self.rank}] Computing loss for cooldown backward")
    #             # Last stage: compute loss and send gradient
    #             target = batch_bwd['label'].to(self.device, dtype=torch.long) if 'label' in batch_bwd else batch_bwd['labels'].to(self.device, dtype=torch.long)
                
    #             loss = self.criterion(output_tensor_bwd, target)
    #             total_loss += loss.item()
    #             print(f"[Rank {self.rank}] Loss computed: {loss.item()}")
    #             with torch.no_grad():
    #                 total_correct += (output_tensor_bwd.argmax(dim=1) == target).sum().item()
    #                 total_samples += target.size(0)
                
    #             scaled_loss = loss / num_micro_batches
    #             scaled_loss.backward()
                
    #             if not self.is_first_stage and input_tensor_bwd is not None:
    #                 if input_tensor_bwd.grad is not None:
    #                     pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
                    
    #         else:
    #             print(f"[Rank {self.rank}] Receiving gradient from next stage for cooldown backward")
    #             # Non-last stages: receive gradient and backprop
    #             grad_buffer = pipeline_recv(self.rank + 1, self.device, self.pp_group, dtype=output_tensor_bwd.dtype)
    #             output_tensor_bwd.backward(gradient=grad_buffer)
    #             print(f"[Rank {self.rank}] Backward pass completed for cooldown")
    #             # Send gradient to previous stage (if not first stage)
    #             if not self.is_first_stage and input_tensor_bwd is not None:
    #                 if input_tensor_bwd.grad is not None:
    #                     print(f"[Rank {self.rank}] Sending gradient to previous stage for cooldown backward")
    #                     pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
    #                     print(f"[Rank {self.rank}] Sent gradient to previous stage for cooldown backward")
            
    #         if self.rank == 0:
    #             pbar.update(1)

    #     if self.rank == 0:
    #         pbar.close()

    #     # ===================================================================
    #     # Finalization
    #     # ===================================================================
        
    #     # Synchronize all ranks before optimizer step
    #     dist.barrier(group=self.pp_group)
        
    #     if self.max_grad_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
    #     self.optimizer.step()
        
    #     if self.is_last_stage:
    #         avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0
    #         accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    #         return avg_loss, accuracy
    #     else:
    #         return 0.0, 0.0

    def train_1f1b_three_phase(self, data_loader, epoch):
        """
        Implements the 1F1B schedule with non-blocking communication.
        
        Three phases:
        1. Warmup: Fill the pipeline with forward passes
        2. Steady State: Interleave forward and backward passes (1F1B)
        3. Cooldown: Drain the pipeline with remaining backward passes
        """
        self.model.train()
        self.optimizer.zero_grad()

        batches = list(data_loader)
        num_micro_batches = len(batches)
        num_steady_microbatches = max(0, num_micro_batches - (self.world_size - 1))
        num_warmup_microbatches = min(self.world_size - self.rank - 1, num_micro_batches)
        num_cooldown_microbatches = max(0, num_micro_batches - num_warmup_microbatches - num_steady_microbatches)

        fwd_inputs_q = deque()
        fwd_outputs_q = deque()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        if self.rank == 0:
            pbar = tqdm(total=num_micro_batches, desc=f"Epoch {epoch} PP-1F1B")

        # ===================================================================
        # PHASE 1: WARMUP - Fill the pipeline
        # ===================================================================
        for i in range(num_warmup_microbatches):
            batch = batches[i]
            if self.is_first_stage:
                input_tensor = batch['image'].to(self.device) if 'image' in batch else batch['images'].to(self.device)
                output_tensor = self.model(input_tensor)
                if not self.is_last_stage:
                    pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
                fwd_inputs_q.append(None)
                fwd_outputs_q.append(output_tensor)
            else:
                input_tensor = pipeline_recv(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
                input_tensor.requires_grad = True
                output_tensor = self.model(input_tensor)
                if not self.is_last_stage:
                    pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
                fwd_inputs_q.append(input_tensor)
                fwd_outputs_q.append(output_tensor)
        
        if self.rank == 0 and num_warmup_microbatches > 0:
            pbar.update(num_warmup_microbatches)

        # ===================================================================
        # PHASE 2: STEADY STATE - 1F1B interleaving
        # ===================================================================
        for i in range(num_steady_microbatches):
            fwd_batch_idx = i + num_warmup_microbatches
            bwd_batch_idx = i

            # FORWARD
            batch_fwd = batches[fwd_batch_idx]
            if self.is_first_stage:
                input_tensor = batch_fwd['image'].to(self.device) if 'image' in batch_fwd else batch_fwd['images'].to(self.device)
                output_tensor = self.model(input_tensor)
                if not self.is_last_stage:
                    pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
                fwd_inputs_q.append(None)
                fwd_outputs_q.append(output_tensor)
            else:
                input_tensor = pipeline_recv(self.rank - 1, self.device, self.pp_group, dtype=torch.float32)
                input_tensor.requires_grad = True
                output_tensor = self.model(input_tensor)
                if not self.is_last_stage:
                    pipeline_send(output_tensor.detach(), self.rank + 1, self.pp_group)
                fwd_inputs_q.append(input_tensor)
                fwd_outputs_q.append(output_tensor)

            # BACKWARD
            batch_bwd = batches[bwd_batch_idx]
            input_tensor_bwd = fwd_inputs_q.popleft()
            output_tensor_bwd = fwd_outputs_q.popleft()
            
            if self.is_last_stage:
                target = batch_bwd['label'].to(self.device, dtype=torch.long) if 'label' in batch_bwd else batch_bwd['labels'].to(self.device, dtype=torch.long)
                loss = self.criterion(output_tensor_bwd, target)
                total_loss += loss.item()
                with torch.no_grad():
                    total_correct += (output_tensor_bwd.argmax(dim=1) == target).sum().item()
                    total_samples += target.size(0)
                scaled_loss = loss / num_micro_batches
                scaled_loss.backward()
                if not self.is_first_stage and input_tensor_bwd is not None:
                    if input_tensor_bwd.grad is not None:
                        pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
            else:
                grad_buffer = pipeline_recv(self.rank + 1, self.device, self.pp_group, dtype=output_tensor_bwd.dtype)
                output_tensor_bwd.backward(gradient=grad_buffer)
                if not self.is_first_stage and input_tensor_bwd is not None:
                    if input_tensor_bwd.grad is not None:
                        pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
            
            if self.rank == 0:
                pbar.update(1)

        # ===================================================================
        # PHASE 3: COOLDOWN - Drain the pipeline
        # ===================================================================
        for i in range(num_cooldown_microbatches):
            bwd_batch_idx = i + num_steady_microbatches
            batch_bwd = batches[bwd_batch_idx]
            input_tensor_bwd = fwd_inputs_q.popleft()
            output_tensor_bwd = fwd_outputs_q.popleft()
            
            if self.is_last_stage:
                target = batch_bwd['label'].to(self.device, dtype=torch.long) if 'label' in batch_bwd else batch_bwd['labels'].to(self.device, dtype=torch.long)
                loss = self.criterion(output_tensor_bwd, target)
                total_loss += loss.item()
                with torch.no_grad():
                    total_correct += (output_tensor_bwd.argmax(dim=1) == target).sum().item()
                    total_samples += target.size(0)
                scaled_loss = loss / num_micro_batches
                scaled_loss.backward()
                if not self.is_first_stage and input_tensor_bwd is not None:
                    if input_tensor_bwd.grad is not None:
                        pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
            else:
                grad_buffer = pipeline_recv(self.rank + 1, self.device, self.pp_group, dtype=output_tensor_bwd.dtype)
                output_tensor_bwd.backward(gradient=grad_buffer)
                if not self.is_first_stage and input_tensor_bwd is not None:
                    if input_tensor_bwd.grad is not None:
                        pipeline_send(input_tensor_bwd.grad, self.rank - 1, self.pp_group)
            
            if self.rank == 0:
                pbar.update(1)

        if self.rank == 0:
            pbar.close()

        # ===================================================================
        # Finalization
        # ===================================================================
        dist.barrier(group=self.pp_group)
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        if self.is_last_stage:
            avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0
            accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return 0.0, 0.0

    def evaluate(self, val_loader):
        """Evaluation with pipeline parallelism."""
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
                        pipeline_send(output, self.rank + 1, self.pp_group)
                else:
                    buffer = pipeline_recv(self.rank - 1, self.device, self.pp_group, dtype=images.dtype)
                    output = self.model(buffer)
                    if not self.is_last_stage:
                        pipeline_send(output, self.rank + 1, self.pp_group)

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