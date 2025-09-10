import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper

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
        self.rank = dist.get_rank()
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

        # Forward pass
        if self.stage_idx == 0:
            # First stage receives input data
            input_data = input_data.to(self.device)
            output = self.model(input_data)
            # Send output to next stage
            Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            return None  # No loss computed at first stage

        elif self.stage_idx == self.num_stages - 1:
            # Last stage receives data from previous stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            input_data = input_data.to(self.device)
            input_data.requires_grad = True
            output = self.model(input_data)
            loss = self.criterion(output, target.to(self.device))
            loss.backward()
            self.optimizer.step()
            return loss.item()
        else:
            # Intermediate stages receive data from previous stage and send to next stage
            input_data = Recv.apply((self.stage_idx - 1) % self.num_stages, self.device.type)
            input_data = input_data.to(self.device)
            input_data.requires_grad = True
            output = self.model(input_data)
            # Send output to next stage
            Send.apply(output, (self.stage_idx + 1) % self.num_stages)
            # output.backward(torch.zeros_like(output))  # Dummy backward to propagate gradients
            # self.optimizer.step()
            return None  # No loss computed at intermediate stages