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
        """
        self.model.train()
        self.optimizer.zero_grad()
        print(f"Rank {self.rank}: Starting train_step...")
        # --- FORWARD PASS ---
        # All stages execute their part of the forward pass.
        # The Send/Recv calls will block and synchronize the stages.
        print("{rank}: is containing which stage? {stage}".format(rank=self.rank, stage=self.stage_idx))
        if self.is_first_stage:
            output = self.model(input_data.to(self.device))
            print(f"Rank {self.rank}: Completed forward pass.")
            if not self.is_last_stage:
                print(f"Rank {self.rank}: Sending data to next stage.")
                Send.apply(output, self.stage_idx + 1, self.pp_group)
                print(f"Rank {self.rank}: Data sent to next stage.")
        elif self.is_last_stage:
            print(f"Rank {self.rank}: Receiving data from previous stage.")
            recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group, input_data.dtype)
            print(f"Rank {self.rank}: Data received from previous stage.")
            recv_data.requires_grad_(True)
            print(f"Rank {self.rank}: Starting last stage forward pass.")
            output = self.model(recv_data)
            print(f"Rank {self.rank}: Completed last stage forward pass.")
        else: # Intermediate stage
            print(f"Rank {self.rank}: Receiving data from previous stage.")
            recv_data = Recv.apply(self.stage_idx - 1, self.device, self.pp_group, input_data.dtype)
            print(f"Rank {self.rank}: Data received from previous stage.")
            print(f"Rank {self.rank}: Starting intermediate stage forward pass.")
            recv_data.requires_grad_(True)
            output = self.model(recv_data)
            print(f"Rank {self.rank}: Completed intermediate stage forward pass.")
            print(f"Rank {self.rank}: Sending data from intermediate stage to next stage.")
            Send.apply(output, self.stage_idx + 1, self.pp_group)
            print(f"Rank {self.rank}: Data sent to next intermediate stage from previous intermediate.")

        # --- BACKWARD PASS ---
        # The backward pass is initiated ONLY on the last stage.
        # This will trigger a chain reaction of gradient calculations and sends
        # back through the pipeline, handled by autograd and our Send/Recv hooks.
        if self.is_last_stage:
            print(f"Rank {self.rank}: Starting backward pass.")
            loss = self.criterion(output, target.to(self.device))
            loss.backward()
            print(f"Rank {self.rank}: Completed backward pass.")
            
        # --- BACKWARD PASS ---
        
        # --- SYNCHRONIZATION AND OPTIMIZER STEP ---
        # There is no 'else' block here. Why?
        # - The last rank is blocked inside `loss.backward()` until the whole chain finishes.
        # - The other ranks are implicitly blocked by the autograd engine, waiting for
        #   their `Send.backward` hook to be called and receive a gradient.
        #
        # Therefore, when the `loss.backward()` call on the last rank finally returns,
        # we are guaranteed that the backward pass has finished EVERYWHERE.
        dist.barrier(group=self.pp_group)
        # Now, ALL ranks have their gradients and can safely update their weights.
        self.optimizer.step()
        print(f"Rank {self.rank}: Optimizer step completed.")
        # Only the last stage has the loss and can calculate accuracy.
        if self.is_last_stage:
            acc = (output_tensor.argmax(dim=1) == target.to(self.device)).float().mean().item()
            return loss.item(), acc * 100
        else:
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