"""
Pipeline Parallel Trainer with AFAB and 1F1B schedules - WITH ACCURACY TRACKING
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from .operations import pipeline_communicate, bidirectional_pipeline_communicate


class PipelineTrainer:
    """
    Trainer class for pipeline parallel training with accuracy tracking.
    """
    def __init__(self, 
                 model, 
                 device_mesh,
                 pp_rank,
                 pp_group, 
                 criterion, 
                 device, 
                 optimizer=None, 
                 max_grad_norm=1.0
                 ):
        """
        Args:
            model: PipelineParallelWrapper instance
            pp_group: Pipeline parallel process group
            criterion: Loss criterion
            device: CUDA device
            optimizer: Optimizer instance
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.pp_group = pp_group
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.is_first_stage = (pp_rank == 0)
        self.is_last_stage = (pp_rank == dist.get_world_size(group=pp_group) - 1)

        self.device_mesh = device_mesh
        self.rank = pp_rank
        self.world_size = dist.get_world_size(group=pp_group)

        # Track metrics during training
        self.batch_labels = []
        self.batch_predictions = []
    
    def train_step_afab(self, data_loader, tensor_shapes, device, dtype):
        """
        All-Forward-All-Backward (AFAB) training step with accuracy tracking.
        
        Returns:
            Tuple of (loss, accuracy) - only on last rank
        """
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        input_tensors, output_tensors = [], []
        grad_acc_steps = data_loader.grad_acc_steps
        
        # Store labels for accuracy calculation
        self.batch_labels = []
        
        # ===== ALL FORWARD PASSES =====
        for _ in range(grad_acc_steps):
            # Receive activation from previous stage
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                device=device,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                dtype=dtype,
                shapes=tensor_shapes
            )
            
            # Get batch from dataloader
            batch = next(data_loader)
            
            # Forward pass
            if self.is_first_stage:
                output_tensor = self.model.forward(batch["images"].to(device))
            else:
                output_tensor = self.model.forward(input_tensor)
            
            # Send activation to next stage
            pipeline_communicate(
                operation='send_forward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                tensor=output_tensor,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Calculate loss and accuracy on last stage
            if self.is_last_stage:
                labels = batch["labels"].to(device)
                loss = self.criterion(output_tensor, labels)
                logging_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                output_tensor = loss
            
            # Save tensors for backward pass
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        # ===== ALL BACKWARD PASSES =====
        for microbatch_idx in range(grad_acc_steps):
            # Receive gradient from next stage
            output_tensor_grad = pipeline_communicate(
                operation='recv_backward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Retrieve saved tensors (FIFO)
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            # Backward pass
            input_tensor_grad = self.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            # Send gradient to previous stage
            pipeline_communicate(
                operation='send_backward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
        
        # Optimizer step
        if self.optimizer is not None:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Return metrics
        if self.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None
    
    def train_step_1f1b(self, data_loader, tensor_shapes, device, dtype):
        """
        1F1B (one-forward-one-backward) training step with accuracy tracking.
        
        Returns:
            Tuple of (loss, accuracy) - only on last rank
        """
        grad_acc_steps = data_loader.grad_acc_steps
        
        # Calculate warmup microbatches
        num_warmup_microbatches = min(
            self.world_size - self.rank - 1,
            grad_acc_steps
        )
        num_microbatches_remaining = grad_acc_steps - num_warmup_microbatches
        
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        input_tensors, output_tensors = [], []
        
        def _forward_step(input_tensor):
            """Helper function for forward pass."""
            batch = next(data_loader)
            
            if self.is_first_stage:
                output_tensor = self.model.forward(batch["images"].to(device))
            else:
                output_tensor = self.model.forward(input_tensor)
            
            # Calculate loss and accuracy on last stage
            if self.is_last_stage:
                labels = batch["labels"].to(device)
                loss = self.criterion(output_tensor, labels)
                
                nonlocal logging_loss, total_correct, total_samples
                logging_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                output_tensor = loss
            
            return output_tensor
        
        # ===== WARMUP PHASE =====
        for _ in range(num_warmup_microbatches):
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            output_tensor = _forward_step(input_tensor)
            pipeline_communicate(
                operation='send_forward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                tensor=output_tensor,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        # ===== STEADY STATE (1F1B) =====
        if num_microbatches_remaining > 0:
            input_tensor = pipeline_communicate(
                operation='recv_forward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
        
        for microbatch_idx in range(num_microbatches_remaining):
            is_last_iteration = (microbatch_idx == num_microbatches_remaining - 1)
            
            # Forward pass
            output_tensor = _forward_step(input_tensor)
            
            # Bidirectional: send forward, receive backward
            output_tensor_grad = bidirectional_pipeline_communicate(
                operation='send_fwd_recv_bwd',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                send_tensor=output_tensor,
                recv_shapes=tensor_shapes,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            
            # Store current tensors
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            
            # Retrieve oldest tensors for backward (FIFO)
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            # Backward pass
            input_tensor_grad = self.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            # Communication for next iteration
            if is_last_iteration:
                input_tensor = None
                pipeline_communicate(
                    operation='send_backward',
                    pp_group=self.pp_group,
                    pp_rank=self.rank,
                    tensor=input_tensor_grad,
                    device=device,
                    dtype=dtype,
                    is_first_stage=self.is_first_stage,
                    is_last_stage=self.is_last_stage,
                    shapes=tensor_shapes
                )
            else:
                # Bidirectional: send backward, receive forward
                input_tensor = bidirectional_pipeline_communicate(
                    operation='send_bwd_recv_fwd',
                    pp_group=self.pp_group,
                    pp_rank=self.rank,
                    send_tensor=input_tensor_grad,
                    recv_shapes=tensor_shapes,
                    device=device,
                    dtype=dtype,
                    is_first_stage=self.is_first_stage,
                    is_last_stage=self.is_last_stage,
                    shapes=tensor_shapes
                )
        
        # ===== COOLDOWN PHASE =====
        for warmup_idx in range(num_warmup_microbatches):
            # Process remaining stored tensors from warmup
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            output_tensor_grad = pipeline_communicate(
                operation='recv_backward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes
            )
            
            input_tensor_grad = self.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            pipeline_communicate(
                operation='send_backward',
                pp_group=self.pp_group,
                pp_rank=self.rank,
                tensor=input_tensor_grad,
                device=device,
                dtype=dtype,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
                shapes=tensor_shapes    
            )
        
        # Optimizer step
        if self.optimizer is not None:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Return metrics
        if self.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None
    
    def evaluate(self, val_loader, tensor_shapes, device, dtype):
        """
        Evaluate the model on validation set with pipeline parallelism.
        
        Args:
            val_loader: Validation DataLoader
            tensor_shapes: Expected tensor shapes for communication
            device: CUDA device
            dtype: Data type
        
        Returns:
            Tuple of (avg_loss, avg_accuracy) - only on last rank
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Receive activation from previous stage
                input_tensor = pipeline_communicate(
                    operation='recv_forward',
                    pp_group=self.pp_group,
                    pp_rank=self.rank,
                    device=device,
                    dtype=dtype,
                    shapes=tensor_shapes
                )
                
                # Forward pass
                if self.is_first_stage:
                    output_tensor = self.model.forward(batch["image"].to(device))
                else:
                    output_tensor = self.model.forward(input_tensor)
                
                # Send activation to next stage
                pipeline_communicate(
                    operation='send_forward',
                    pp_group=self.pp_group,
                    pp_rank=self.rank,
                    tensor=output_tensor,
                    device=device,
                    dtype=dtype,
                    shapes=tensor_shapes
                )
                
                # Calculate metrics on last stage
                if self.is_last_stage:
                    labels = batch["label"].to(device)
                    loss = self.criterion(output_tensor, labels)
                    
                    total_loss += loss.item() * labels.size(0)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output_tensor, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
        
        # Return metrics
        if self.is_last_stage and total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = (total_correct / total_samples) * 100.0
            return avg_loss, avg_accuracy
        else:
            return None, None
