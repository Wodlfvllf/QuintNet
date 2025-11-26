"""
Pipeline Parallel Trainer

This module defines the `PipelineTrainer` class, which is responsible for
executing a single training step for a pipeline-parallel model. It handles
the complexities of micro-batching, communication between pipeline stages,
and different pipeline schedules (e.g., 1F1B, AFAB).

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The `PipelineTrainer` is a low-level component used by the high-level `Trainer`
class when pipeline parallelism is active. Its primary role is to manage the
forward and backward passes across pipeline stages for a single optimization
step, including gradient accumulation.

It delegates the actual micro-batch scheduling to specialized schedule classes
(e.g., `OneFOneBSchedule`, `AllFwdAllBwdSchedule`).

===============================================================================
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from ...core import pipeline_communicate, bidirectional_pipeline_communicate
from .schedule import AllFwdAllBwdSchedule, OneFOneBSchedule


class PipelineTrainer:
    """
    Trainer class for executing a single training step with pipeline parallelism.

    This class manages the forward and backward passes for micro-batches across
    pipeline stages, handling inter-stage communication and gradient accumulation.
    It supports different pipeline schedules.
    """
    def __init__(self, 
                 model, 
                 device_mesh,
                 pp_rank,
                 pp_group, 
                 criterion, 
                 device, 
                 optimizer=None, 
                 max_grad_norm=1.0,
                 schedule_type='1f1b'
                 ):
        """
        Initializes the PipelineTrainer.

        Args:
            model: The `PipelineParallelWrapper` instance representing the local
                stage of the model.
            device_mesh: The device mesh object, used for global rank information.
            pp_rank (int): The rank of the current process within the pipeline
                parallel group.
            pp_group (dist.ProcessGroup): The process group for pipeline
                parallel communication.
            criterion: The loss function (e.g., `nn.CrossEntropyLoss`).
            device: The CUDA device for the current process.
            optimizer (torch.optim.Optimizer, optional): The optimizer instance.
            max_grad_norm (float, optional): Maximum gradient norm for clipping.
            schedule_type (str, optional): The type of pipeline schedule to use
                ('1f1b' for One-Forward-One-Backward or 'afab' for All-Forward-All-Backward).
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

        # Track metrics during training (used by schedule classes)
        self.batch_labels = []
        self.batch_predictions = []

        # Initialize the chosen pipeline schedule
        if schedule_type == '1f1b':
            self.schedule = OneFOneBSchedule(self)
        elif schedule_type == 'afab':
            self.schedule = AllFwdAllBwdSchedule(self)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        """
        Executes a single training step using the configured pipeline schedule.

        This method delegates the actual micro-batch processing and communication
        to the selected schedule (1F1B or AFAB).

        Args:
            data_loader: The `PipelineDataLoader` providing micro-batches.
            tensor_shapes (tuple): The expected shapes of tensors communicated
                between pipeline stages.
            device (torch.device): The CUDA device.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple of (loss, accuracy) - only on the last rank of the pipeline.
        """
        return self.schedule.train_step(data_loader, tensor_shapes, device, dtype)
    
    def evaluate(self, val_loader, tensor_shapes, device, dtype):
        """
        Evaluates the model on a validation set with pipeline parallelism.

        This method performs forward passes for validation micro-batches across
        pipeline stages and aggregates metrics on the last stage.

        Args:
            val_loader: The DataLoader for the validation set.
            tensor_shapes (tuple): Expected tensor shapes for communication.
            device (torch.device): The CUDA device.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple of (avg_loss, avg_accuracy) - only on the last rank of the pipeline.
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
                
                # Forward pass through the local stage
                if self.is_first_stage:
                    # First stage takes image data
                    output_tensor = self.model.forward(batch["image"].to(device))
                else:
                    # Subsequent stages take activations from previous stage
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
                
                # Calculate metrics on the last stage only
                if self.is_last_stage:
                    labels = batch["label"].to(device)
                    loss = self.criterion(output_tensor, labels)
                    
                    total_loss += loss.item() * labels.size(0)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output_tensor, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
        
        # Return metrics only from the last stage
        if self.is_last_stage and total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = (total_correct / total_samples) * 100.0
            return avg_loss, avg_accuracy
        else:
            return None, None