"""
Pipeline Parallel Trainer with AFAB and 1F1B schedules - WITH ACCURACY TRACKING
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from QuintNet.core.communication import pipeline_communicate, bidirectional_pipeline_communicate
from QuintNet.parallelism.pipeline_parallel.schedule import AllFwdAllBwdSchedule, OneFOneBSchedule


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
                 max_grad_norm=1.0,
                 schedule_type='1f1b'
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

        if schedule_type == '1f1b':
            self.schedule = OneFOneBSchedule(self)
        elif schedule_type == 'afab':
            self.schedule = AllFwdAllBwdSchedule(self)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        """
        All-Forward-All-Backward (AFAB) training step with accuracy tracking.
        
        Returns:
            Tuple of (loss, accuracy) - only on last rank
        """
        return self.schedule.train_step(data_loader, tensor_shapes, device, dtype)
    
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
