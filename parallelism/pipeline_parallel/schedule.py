"""
Pipeline Schedules

This module defines the abstract base class for pipeline schedules and provides
concrete implementations for common schedules like 1F1B (One-Forward-One-Backward)
and AFAB (All-Forward-All-Backward). These schedules dictate the order and
communication patterns of micro-batches between pipeline stages during training.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Pipeline parallelism requires breaking down a global batch into smaller
micro-batches. These micro-batches are then processed by the pipeline stages
in a specific order to maximize GPU utilization and minimize pipeline bubbles.

- **1F1B (One-Forward-One-Backward):** After a certain number of micro-batches
  have performed their forward pass (warm-up phase), each time a micro-batch
  completes its forward pass, another micro-batch performs its backward pass.
  This keeps the pipeline full.

- **AFAB (All-Forward-All-Backward):** All micro-batches complete their forward
  passes before any micro-batch begins its backward pass. This is simpler to
  implement but generally less efficient than 1F1B.

The schedule classes are closely integrated with the `PipelineTrainer` (which
holds the model, optimizer, etc.) to execute these communication and computation
patterns.

===============================================================================
"""

import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from ...core.communication import pipeline_communicate, bidirectional_pipeline_communicate

class PipelineSchedule(ABC):
    """
    Abstract base class for all pipeline schedules.

    All concrete pipeline schedule implementations must inherit from this class
    and implement the `train_step` method.
    """
    def __init__(self, trainer):
        """
        Initializes the PipelineSchedule.

        Args:
            trainer: An instance of `PipelineTrainer` (or a class with a
                compatible interface) that holds the model, optimizer, etc.
                This allows the schedule to access necessary training components.
        """
        self.trainer = trainer

    @abstractmethod
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        """
        Abstract method to execute a single training step for a pipeline schedule.

        This method defines the specific sequence of forward and backward passes
        and inter-stage communication for micro-batches.

        Args:
            data_loader: The `PipelineDataLoader` providing micro-batches.
            tensor_shapes (tuple): The expected shapes of tensors communicated
                between pipeline stages.
            device (torch.device): The CUDA device.
            dtype (torch.dtype): The data type of the tensors.
        """
        pass

class AllFwdAllBwdSchedule(PipelineSchedule):
    """
    Implements the All-Forward-All-Backward (AFAB) pipeline schedule.

    In the AFAB schedule, all micro-batches complete their forward passes
    before any micro-batch begins its backward pass. This is simpler
    but can lead to pipeline bubbles (idle times).
    """
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        """
        Executes a single training step using the AFAB schedule.

        All micro-batches perform their forward passes. Then, all
        micro-batches perform their backward passes.

        Args:
            data_loader: The `PipelineDataLoader` providing micro-batches.
            tensor_shapes (tuple): The expected shapes of tensors communicated
                between pipeline stages.
            device (torch.device): The CUDA device.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple of (loss, accuracy) - only on the last rank of the pipeline.
        """
        trainer = self.trainer
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        input_tensors, output_tensors = [], [] # Stores activations for backward pass
        grad_acc_steps = data_loader.grad_acc_steps
        
        # ===== ALL FORWARD PASSES =====
        for _ in range(grad_acc_steps):
            # Receive activation from the previous stage
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                device=device,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                dtype=dtype,
                shapes=tensor_shapes
            )
            
            # Get a micro-batch from the data loader
            batch = next(data_loader)
            
            # Perform forward pass
            if trainer.is_first_stage:
                # First stage uses image data as input
                output_tensor = trainer.model.forward(batch["images"].to(device))
            else:
                # Subsequent stages use activations from previous stage
                output_tensor = trainer.model.forward(input_tensor)
            
            # Send activation to the next stage
            pipeline_communicate(
                operation='send_forward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                tensor=output_tensor,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            
            # If this is the last stage, calculate loss and accuracy
            if trainer.is_last_stage:
                labels = batch["labels"].to(device)
                loss = trainer.criterion(output_tensor, labels)
                logging_loss += loss.item()
                
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # The loss scalar is used as input gradient for the backward pass
                output_tensor = loss
            
            # Store inputs and outputs for the corresponding backward pass
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        # ===== ALL BACKWARD PASSES =====
        for microbatch_idx in range(grad_acc_steps):
            # Receive gradient from the next stage
            output_tensor_grad = pipeline_communicate(
                operation='recv_backward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Retrieve saved inputs and outputs from the forward pass (FIFO order)
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            # Perform backward pass through the local stage
            # The model's backward method expects saved input, output, and incoming gradient
            input_tensor_grad = trainer.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            # Send gradients to the previous stage
            pipeline_communicate(
                operation='send_backward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
        
        # Perform optimizer step after all gradients have been accumulated
        if trainer.optimizer is not None:
            if trainer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), 
                    trainer.max_grad_norm
                )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        
        # Return metrics (only calculated/meaningful on the last stage)
        if trainer.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None

class OneFOneBSchedule(PipelineSchedule):
    """
    Implements the 1F1B (One-Forward-One-Backward) pipeline schedule.

    In the 1F1B schedule, after a warm-up phase where only forward passes occur,
    an equal number of forward and backward passes are performed concurrently.
    This aims to keep the GPUs as busy as possible, significantly reducing
    pipeline bubbles.
    """
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        """
        Executes a single training step using the 1F1B schedule.

        Args:
            data_loader: The `PipelineDataLoader` providing micro-batches.
            tensor_shapes (tuple): The expected shapes of tensors communicated
                between pipeline stages.
            device (torch.device): The CUDA device.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple of (loss, accuracy) - only on the last rank of the pipeline.
        """
        trainer = self.trainer
        grad_acc_steps = data_loader.grad_acc_steps
        
        # Calculate the number of micro-batches for warm-up and steady state
        # The warm-up phase involves only forward passes to fill the pipeline.
        num_warmup_microbatches = min(
            trainer.world_size - trainer.rank - 1,
            grad_acc_steps
        )
        # The remaining micro-batches are processed in the steady state (1F1B)
        num_microbatches_remaining = grad_acc_steps - num_warmup_microbatches
        
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Queues to store activations/outputs for corresponding backward passes
        input_tensors, output_tensors = [], []
        
        def _forward_step(input_tensor):
            """Helper function for performing a micro-batch forward pass."""
            rank = dist.get_rank()
            # print(f"[Rank {rank}] _forward_step: Getting batch", flush=True)
            batch = next(data_loader)
            # print(f"[Rank {rank}] _forward_step: Got batch", flush=True)
            
            if trainer.is_first_stage:
                print(f"[Rank {rank}] _forward_step: Calling model.forward (Stage 0)", flush=True)
                output_tensor = trainer.model.forward(batch["images"].to(device))
                print(f"[Rank {rank}] _forward_step: Finished model.forward (Stage 0)", flush=True)
            else:
                print(f"[Rank {rank}] _forward_step: Calling model.forward (Stage > 0)", flush=True)
                output_tensor = trainer.model.forward(input_tensor)
                print(f"[Rank {rank}] _forward_step: Finished model.forward (Stage > 0)", flush=True)
            
            # On the last stage, calculate loss and track accuracy
            if trainer.is_last_stage:
                labels = batch["labels"].to(device)
                loss = trainer.criterion(output_tensor, labels)
                
                # nonlocal is used to modify variables in the enclosing scope
                nonlocal logging_loss, total_correct, total_samples
                logging_loss += loss.item()
                
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                output_tensor = loss # The loss scalar is used as input gradient for the backward pass
            
            return output_tensor
        
        # ===== WARMUP PHASE =====
        # Only forward passes are executed to fill the pipeline.
        rank = dist.get_rank()
        print(f"[Rank {rank}] 1F1B: START Warmup Phase ({num_warmup_microbatches} steps)", flush=True)
        for i in range(num_warmup_microbatches):
            print(f"[Rank {rank}] 1F1B: Warmup step {i}", flush=True)
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            output_tensor = _forward_step(input_tensor)
            pipeline_communicate(
                operation='send_forward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                tensor=output_tensor,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        print(f"[Rank {rank}] 1F1B: END Warmup Phase", flush=True)
        
        # ===== STEADY STATE (1F1B) =====
        # Forward and backward passes happen concurrently.
        # This phase starts by receiving a forward activation from the previous stage (if any).
        print(f"[Rank {rank}] 1F1B: START Steady State ({num_microbatches_remaining} steps)", flush=True)
        if num_microbatches_remaining > 0:
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
        
        for microbatch_idx in range(num_microbatches_remaining):
            print(f"[Rank {rank}] 1F1B: Steady State step {microbatch_idx}", flush=True)
            is_last_iteration = (microbatch_idx == num_microbatches_remaining - 1)
            
            # Forward pass for current micro-batch
            output_tensor = _forward_step(input_tensor)
            
            # Bidirectional communication: send forward activation, receive backward gradient
            output_tensor_grad = bidirectional_pipeline_communicate(
                operation='send_fwd_recv_bwd',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                send_tensor=output_tensor,
                recv_shapes=tensor_shapes,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Store current micro-batch's inputs/outputs for its backward pass
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            
            # Retrieve the oldest micro-batch's inputs/outputs for its backward pass (FIFO)
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            # Perform backward pass for the oldest micro-batch
            input_tensor_grad = trainer.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            # Handle communication for the next iteration:
            if is_last_iteration:
                # If this is the last micro-batch, only send the final gradients backward.
                input_tensor = None
                pipeline_communicate(
                    operation='send_backward',
                    pp_group=trainer.pp_group,
                    pp_rank=trainer.rank,
                    tensor=input_tensor_grad,
                    device=device,
                    dtype=dtype,
                    is_first_stage=trainer.is_first_stage,
                    is_last_stage=trainer.is_last_stage,
                    shapes=tensor_shapes
                )
            else:
                # Otherwise, send backward gradient and receive next forward activation concurrently.
                input_tensor = bidirectional_pipeline_communicate(
                    operation='send_bwd_recv_fwd',
                    pp_group=trainer.pp_group,
                    pp_rank=trainer.rank,
                    send_tensor=input_tensor_grad,
                    recv_shapes=tensor_shapes,
                    device=device,
                    dtype=dtype,
                    is_first_stage=trainer.is_first_stage,
                    is_last_stage=trainer.is_last_stage,
                    shapes=tensor_shapes
                )
        print(f"[Rank {rank}] 1F1B: END Steady State", flush=True)
        
        # ===== COOLDOWN PHASE =====
        # Only backward passes are executed to clear the pipeline.
        print(f"[Rank {rank}] 1F1B: START Cooldown Phase", flush=True)
        for warmup_idx in range(num_warmup_microbatches):
            print(f"[Rank {rank}] 1F1B: Cooldown step {warmup_idx}", flush=True)
            # Retrieve remaining stored inputs and outputs
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            # Receive gradient from the next stage
            output_tensor_grad = pipeline_communicate(
                operation='recv_backward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Perform backward pass
            input_tensor_grad = trainer.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            pipeline_communicate(
                operation='send_backward',
                pp_group=trainer.pp_group,
                pp_rank=trainer.rank,
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
                is_first_stage=trainer.is_first_stage,
                is_last_stage=trainer.is_last_stage,
                shapes=tensor_shapes    
            )
        print(f"[Rank {rank}] 1F1B: END Cooldown Phase", flush=True)
        
        # Perform optimizer step after all gradients have been accumulated
        if trainer.optimizer is not None:
            if trainer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), 
                    trainer.max_grad_norm
                )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        
        # Return metrics (only calculated/meaningful on the last stage)
        if trainer.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None
