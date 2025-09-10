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
        self.model = PipelineParallelWrapper(model, pp_group).to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size  # assume world_size == num_stages
        self.stage_idx = self.rank  # direct mapping

    def train_step(self, input_data, target):
        """
        Perform a single training step with pipeline parallelism.
        input_data: input tensor for the first stage (only for rank 0)
        target: target tensor for loss computation (only for last stage)
        """
        self.model.train()
        self.optimizer.zero_grad()
        # print(self.stage_idx, self.num_stages, self.rank, self.world_size)
        # Forward pass
        if self.stage_idx == 0:
            # First stage receives input data
            print("rank - ", self.rank, " input - ", 'images shape - ', input_data.shape)
            
            input_data = input_data.to(self.device)
            output = self.model(input_data)
            print("rank - ", self.rank, " output - ", 'images shape - ', output.shape)

            # Send output to next stage
            Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            return None, None  # No loss computed at first stage

        elif self.stage_idx == self.num_stages-1:
            # Last stage receives data from previous stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            print("rank - ", self.rank, " input - ", 'images shape - ', input_data.shape)
            
            input_data = input_data.to(self.device)
            input_data.requires_grad = True
            output = self.model(input_data)
            print("rank - ", self.rank, " output - ", 'images shape - ', output.shape)
            
            loss = self.criterion(output, target.to(self.device))
            loss.backward()
            self.optimizer.step()
            acc = (output.argmax(dim=1) == target.to(self.device)).float().mean().item()
            return loss.item(), acc
        else:
            # Intermediate stages receive data from previous stage and send to next stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            print("rank - ", self.rank, " input - ", 'images shape - ', input_data.shape)

            input_data = input_data.to(self.device)
            input_data.requires_grad = True
            output = self.model(input_data)
            print("rank - ", self.rank, " output - ", 'images shape - ', output.shape)
            
            # Send output to next stage
            Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            # output.backward(torch.zeros_like(output))  # Dummy backward to propagate gradients
            # self.optimizer.step()
            return None, None  # No loss computed at intermediate stages
        
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
                    print("rank - ", self.rank, " stage - ", self.stage_idx, 'images shape - ', images.shape)
                    output = self.model(images.to(self.device))
                    print("rank - ", self.rank, " stage - ", self.stage_idx, 'images shape - ', output.shape)
                    
                    # Send output to next stage
                    Send.apply(output, (self.stage_idx + 1) % self.num_stages)
                    continue  # No loss computed at first stage

                elif self.stage_idx == self.num_stages - 1:
                    # Last stage receives data from previous stage
                    data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
                    print("rank - ", self.rank, " stage - ", self.stage_idx, 'images shape - ', data.shape)
                    data = data.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target.to(self.device))
                    total_loss += loss.item() * data.size(0)
                    total_correct += (output.argmax(dim=1) == target.to(self.device)).sum().item()
                    total_samples += data.size(0)

                else:
                    # Intermediate stages receive data from previous stage and send to next stage
                    data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
                    print("rank - ", self.rank, " stage - ", self.stage_idx, 'images shape - ', data.shape)
                    data = data.to(self.device)
                    output = self.model(data)
                    print("rank - ", self.rank, " stage - ", self.stage_idx, 'images shape - ', output.shape)
                    
                    # Send output to next stage
                    Send.apply(output, (self.stage_idx + 1) % self.num_stages)
                    continue  # No loss computed at intermediate stages

        if self.stage_idx == self.num_stages - 1:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            return avg_loss, accuracy
        else:
            return None, None  # Only last stage computes and returns loss and accuracy