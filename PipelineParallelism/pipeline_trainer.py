import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper
from .operations import Send, Recv

class PipelineTrainer:
    def __init__(self, model: nn.ModuleList, pp_group, optimizer, criterion, device):
        """
        model: torch.nn.ModuleList of layers/blocks (must be ModuleList for easy slicing)
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
        
        # Initialize model wrapper and move to device
        self.model = PipelineParallelWrapper(model, pp_group).to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        
        print(f"Initialized PipelineTrainer - Rank: {self.rank}, Stage: {self.stage_idx}")

    def train_step(self, input_data, target):
        """
        Perform a single training step with pipeline parallelism.
        input_data: input tensor for the first stage (only meaningful for rank 0)
        target: target tensor for loss computation (only meaningful for last stage)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.stage_idx == 0:
            # First stage receives input data
            print(f"Rank {self.rank}: Processing input with shape {input_data.shape}")
            
            input_data = input_data.to(self.device)
            output = self.model(input_data)
            print(f"Rank {self.rank}: Output shape {output.shape}")

            # Send output to next stage if not last stage
            if self.num_stages > 1:
                Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            else:
                # Single stage case - compute loss here
                loss = self.criterion(output, target.to(self.device))
                loss.backward()
                self.optimizer.step()
                acc = (output.argmax(dim=1) == target.to(self.device)).float().mean().item()
                return loss.item(), acc
            
            return None, None

        elif self.stage_idx == self.num_stages - 1:
            # Last stage receives data from previous stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            print(f"Rank {self.rank}: Received input with shape {input_data.shape}")
            
            input_data = input_data.to(self.device)
            # Enable gradients for backward pass
            input_data.requires_grad_(True)
            
            output = self.model(input_data)
            print(f"Rank {self.rank}: Output shape {output.shape}")
            
            # Compute loss and backward pass
            target = target.to(self.device)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Send gradients backward (this should be handled by your Send/Recv autograd)
            # The gradient of input_data will be sent back automatically
            
            self.optimizer.step()
            acc = (output.argmax(dim=1) == target).float().mean().item()
            return loss.item(), acc
            
        else:
            # Intermediate stages receive data from previous stage and send to next stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            print(f"Rank {self.rank}: Received input with shape {input_data.shape}")

            input_data = input_data.to(self.device)
            input_data.requires_grad_(True)
            
            output = self.model(input_data)
            print(f"Rank {self.rank}: Output shape {output.shape}")
            
            # Send output to next stage
            Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            
            # For intermediate stages, we need to wait for backward pass
            # This should be handled automatically by the autograd system
            self.optimizer.step()
            return None, None
        
    def evaluate(self, val_loader):
        """
        Evaluate the model on the validation set.
        val_loader: DataLoader for validation data
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch['image'], batch['label']
                
                if self.stage_idx == 0:
                    # First stage receives input data
                    print(f"Eval - Rank {self.rank}: Input shape {images.shape}")
                    images = images.to(self.device)
                    output = self.model(images)
                    print(f"Eval - Rank {self.rank}: Output shape {output.shape}")
                    
                    # Send output to next stage if not last stage
                    if self.num_stages > 1:
                        Send.apply(output, (self.stage_idx + 1) % self.num_stages)
                    else:
                        # Single stage case
                        targets = targets.to(self.device)
                        loss = self.criterion(output, targets)
                        total_loss += loss.item() * images.size(0)
                        total_correct += (output.argmax(dim=1) == targets).sum().item()
                        total_samples += images.size(0)
                    continue

                elif self.stage_idx == self.num_stages - 1:
                    # Last stage receives data from previous stage
                    data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
                    print(f"Eval - Rank {self.rank}: Received shape {data.shape}")
                    
                    data = data.to(self.device)
                    output = self.model(data)
                    
                    targets = targets.to(self.device)
                    loss = self.criterion(output, targets)
                    total_loss += loss.item() * data.size(0)
                    total_correct += (output.argmax(dim=1) == targets).sum().item()
                    total_samples += data.size(0)

                else:
                    # Intermediate stages receive data from previous stage and send to next stage
                    data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
                    print(f"Eval - Rank {self.rank}: Received shape {data.shape}")
                    
                    data = data.to(self.device)
                    output = self.model(data)
                    print(f"Eval - Rank {self.rank}: Output shape {output.shape}")
                    
                    # Send output to next stage
                    Send.apply(output, (self.stage_idx + 1) % self.num_stages)
                    continue

        # Return results only from last stage (or single stage)
        if self.stage_idx == self.num_stages - 1 or self.num_stages == 1:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return None, None