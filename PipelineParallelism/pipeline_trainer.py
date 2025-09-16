import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager
from .pp_wrapper import PipelineParallelWrapper
from .operations import Send, Recv


class PipelineTrainer:
    """Pipeline trainer with manual backward pass management"""
    
    def __init__(self, model, pp_group, optimizer, criterion, device):
        self.pp_group = pp_group
        self.device = device
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size
        self.stage_idx = self.rank

        if isinstance(model, PipelineParallelWrapper):
            self.model = model
        else:
            self.model = PipelineParallelWrapper(model, pp_group).to(device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.is_first_stage = (self.stage_idx == 0)
        self.is_last_stage = (self.stage_idx == self.num_stages - 1)

        # To store tensors for backward pass
        self.saved_input = None
        self.saved_output = None

        print(f"PipelineTrainerManual initialized - Rank: {self.rank}, Stage: {self.stage_idx}")

    def train_step(self, input_data, target):
        self.model.train()
        self.optimizer.zero_grad()

        # --- FORWARD PASS ---
        if self.is_first_stage:
            self.saved_input = input_data.to(self.device)
            self.saved_input.requires_grad_()
            output = self.model(self.saved_input)
            if not self.is_last_stage:
                # Use Send but detach to avoid autograd tracking
                Send.apply(output.detach(), self.stage_idx + 1, self.pp_group)
        else:
            # Receive data from previous stage
            recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group, input_data.dtype)
            self.saved_input = recv_data
            self.saved_input.requires_grad_()
            # print("before last stage input shape is ", self.saved_input.shape)
            output = self.model(self.saved_input)
            # print("after last stage output shape is ", output.shape)
            if not self.is_last_stage:
                # Use Send but detach to avoid autograd tracking
                Send.apply(output.detach(), self.stage_idx + 1, self.pp_group)
        
        self.saved_output = output

        # --- MANUAL BACKWARD PASS ---
        if self.is_last_stage:
            # Compute loss and backward
            # loss = self.criterion(self.saved_output, target.to(self.device).unsqueeze(-1))
            print(self.saved_output.shape, target.shape)
            loss = self.criterion(self.saved_output, target.to(device=self.device, dtype=torch.long))
            loss.backward()
            
            # Manually send gradient to previous stage
            if not self.is_first_stage:
                grad_to_send = self.saved_input.grad.contiguous()
                dist.send(tensor=grad_to_send, dst=self.stage_idx - 1, group=self.pp_group)
        else:
            # Manually receive gradient from next stage
            grad_shape = self.saved_output.shape
            grad_dtype = self.saved_output.dtype
            received_grad = torch.zeros(grad_shape, dtype=grad_dtype, device=self.device)
            dist.recv(tensor=received_grad, src=self.stage_idx + 1, group=self.pp_group)
            
            # Perform backward pass with received gradient
            self.saved_output.backward(gradient=received_grad)

            if not self.is_first_stage:
                # Manually send gradient to previous stage
                grad_to_send = self.saved_input.grad.contiguous()
                dist.send(tensor=grad_to_send, dst=self.stage_idx - 1, group=self.pp_group)

        # --- OPTIMIZER STEP ---
        self.optimizer.step()

        if self.is_last_stage:
            acc = (self.saved_output.argmax(dim=1) == target.to(self.device)).float().mean().item()
            return loss.item(), acc * 100
        else:
            return None, None