import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from QuintNet.core.communication import pipeline_communicate, bidirectional_pipeline_communicate

class PipelineSchedule(ABC):
    """
    Abstract base class for pipeline schedules.
    """
    def __init__(self, trainer):
        self.trainer = trainer

    @abstractmethod
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        pass

class AllFwdAllBwdSchedule(PipelineSchedule):
    """
    All-Forward-All-Backward (AFAB) schedule.
    """
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        # This is the logic from train_step_afab
        trainer = self.trainer
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        input_tensors, output_tensors = [], []
        grad_acc_steps = data_loader.grad_acc_steps
        
        trainer.batch_labels = []
        
        # ===== ALL FORWARD PASSES =====
        for _ in range(grad_acc_steps):
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
            
            batch = next(data_loader)
            
            if trainer.is_first_stage:
                output_tensor = trainer.model.forward(batch["images"].to(device))
            else:
                output_tensor = trainer.model.forward(input_tensor)
            
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
            
            if trainer.is_last_stage:
                labels = batch["labels"].to(device)
                loss = trainer.criterion(output_tensor, labels)
                logging_loss += loss.item()
                
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                output_tensor = loss
            
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        # ===== ALL BACKWARD PASSES =====
        for microbatch_idx in range(grad_acc_steps):
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
            
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
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
        
        if trainer.optimizer is not None:
            if trainer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), 
                    trainer.max_grad_norm
                )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        
        if trainer.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None

class OneFOneBSchedule(PipelineSchedule):
    """
    1F1B (one-forward-one-backward) schedule.
    """
    def train_step(self, data_loader, tensor_shapes, device, dtype):
        # This is the logic from train_step_1f1b
        trainer = self.trainer
        grad_acc_steps = data_loader.grad_acc_steps
        
        num_warmup_microbatches = min(
            trainer.world_size - trainer.rank - 1,
            grad_acc_steps
        )
        num_microbatches_remaining = grad_acc_steps - num_warmup_microbatches
        
        logging_loss = 0.0
        total_correct = 0
        total_samples = 0
        input_tensors, output_tensors = [], []
        
        def _forward_step(input_tensor):
            batch = next(data_loader)
            
            if trainer.is_first_stage:
                output_tensor = trainer.model.forward(batch["images"].to(device))
            else:
                output_tensor = trainer.model.forward(input_tensor)
            
            if trainer.is_last_stage:
                labels = batch["labels"].to(device)
                loss = trainer.criterion(output_tensor, labels)
                
                nonlocal logging_loss, total_correct, total_samples
                logging_loss += loss.item()
                
                _, predicted = torch.max(output_tensor, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                output_tensor = loss
            
            return output_tensor
        
        # ===== WARMUP PHASE =====
        for _ in range(num_warmup_microbatches):
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
        
        # ===== STEADY STATE (1F1B) =====
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
            is_last_iteration = (microbatch_idx == num_microbatches_remaining - 1)
            
            output_tensor = _forward_step(input_tensor)
            
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
            
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
            input_tensor_grad = trainer.model.backward(
                input_tensor, 
                output_tensor, 
                output_tensor_grad
            )
            
            if is_last_iteration:
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
        
        # ===== COOLDOWN PHASE =====
        for warmup_idx in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            
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
        
        if trainer.optimizer is not None:
            if trainer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), 
                    trainer.max_grad_norm
                )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        
        if trainer.is_last_stage:
            avg_loss = logging_loss / grad_acc_steps
            avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            return avg_loss, avg_accuracy
        else:
            return None, None