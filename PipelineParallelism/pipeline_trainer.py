import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper
from .operations import Send, Recv

class PipelineTrainer:
    def __init__(self, model, pp_group, optimizer, criterion, device):
        """
        model: The model (already wrapped with PipelineParallelWrapper)
        pp_group: process group for pipeline parallelism
        optimizer: optimizer for training
        criterion: loss function
        device: device to run the model on
        """
        self.pp_group = pp_group
        self.device = device
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size
        self.stage_idx = self.rank
        
        # Store model (should already be PipelineParallelWrapper)
        if isinstance(model, PipelineParallelWrapper):
            self.model = model
        else:
            # If not wrapped, wrap it
            self.model = PipelineParallelWrapper(model, pp_group).to(device)
        
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Flags for stage position
        self.is_first_stage = (self.stage_idx == 0)
        self.is_last_stage = (self.stage_idx == self.num_stages - 1)
        
        print(f"PipelineTrainer initialized - Rank: {self.rank}, Stage: {self.stage_idx}")

    def train_step(self, input_data, target):
        """
        Perform a single training step with pipeline parallelism.
        Uses simple forward-backward scheduling (no pipelining of microbatches).
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.is_first_stage:
            # First stage: process input and send to next
            input_data = input_data.to(self.device)
            output = self.model(input_data)
            
            if not self.is_last_stage:  # Multi-stage case
                # Send output to next stage
                Send.apply(output, self.stage_idx + 1, self.pp_group)
                # First stage waits for gradient from next stage in backward pass
                # The Send.backward will receive gradients automatically
            else:  # Single stage case
                # Compute loss and backward
                loss = self.criterion(output, target.to(self.device))
                loss.backward()
                self.optimizer.step()
                acc = (output.argmax(dim=1) == target.to(self.device)).float().mean().item()
                return loss.item(), acc * 100
                
        elif self.is_last_stage:
            # Last stage: receive from previous, compute loss, start backward
            # Receive activations from previous stage
            recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group)
            recv_data = recv_data.to(self.device)
            recv_data.requires_grad_(True)
            
            # Forward through last stage
            output = self.model(recv_data)
            
            # Compute loss
            target = target.to(self.device)
            loss = self.criterion(output, target)
            
            # Backward pass - this will send gradients back through Recv.backward
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            acc = (output.argmax(dim=1) == target).float().mean().item()
            return loss.item(), acc * 100
            
        else:
            # Intermediate stage: receive, forward, send
            # Receive from previous stage
            recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group)
            recv_data = recv_data.to(self.device)
            recv_data.requires_grad_(True)
            
            # Forward through this stage
            output = self.model(recv_data)
            
            # Send to next stage
            Send.apply(output, self.stage_idx + 1, self.pp_group)
            
            # The backward pass will be triggered by gradients from next stage
            # coming through Send.backward, and will send gradients to previous
            # stage through Recv.backward
            
        # For non-last stages, we need to wait for the backward pass to complete
        # This happens automatically through autograd when the last stage calls backward()
        
        # Update parameters after backward pass completes
        if not self.is_last_stage:
            self.optimizer.step()
            
        return None, None

    def evaluate(self, val_loader):
        """
        Evaluate the model on validation set.
        Only the last stage computes metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image']
                targets = batch['label']
                
                if self.is_first_stage:
                    # First stage: process input
                    images = images.to(self.device)
                    output = self.model(images)
                    
                    if not self.is_last_stage:
                        # Send to next stage
                        Send.apply(output, self.stage_idx + 1, self.pp_group)
                    else:
                        # Single stage case
                        targets = targets.to(self.device)
                        loss = self.criterion(output, targets)
                        total_loss += loss.item() * images.size(0)
                        total_correct += (output.argmax(dim=1) == targets).sum().item()
                        total_samples += images.size(0)
                        
                elif self.is_last_stage:
                    # Last stage: receive and compute metrics
                    recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group)
                    recv_data = recv_data.to(self.device)
                    
                    output = self.model(recv_data)
                    
                    targets = targets.to(self.device)
                    loss = self.criterion(output, targets)
                    total_loss += loss.item() * recv_data.size(0)
                    total_correct += (output.argmax(dim=1) == targets).sum().item()
                    total_samples += recv_data.size(0)
                    
                else:
                    # Intermediate stage: receive and forward
                    recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group)
                    recv_data = recv_data.to(self.device)
                    
                    output = self.model(recv_data)
                    
                    # Send to next stage
                    Send.apply(output, self.stage_idx + 1, self.pp_group)
        
        # Return metrics only from last stage
        if self.is_last_stage or self.num_stages == 1:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return None, None