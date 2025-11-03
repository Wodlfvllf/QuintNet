"""
Pipeline Parallel Wrapper for Vision Transformer.
Splits model across pipeline stages.
"""

import torch
import torch.nn as nn


class PipelineParallelWrapper(nn.Module):
    """
    Wraps a Vision Transformer model for pipeline parallelism.
    Distributes transformer blocks across multiple GPUs.
    """
    def __init__(self, 
                 model, 
                 device_mesh,
                 rank,
                 pp_group,
                 pp_size,
                 device
                 ):
        """
        Args:
            model: The complete ViT model to be split
            pgm: ProcessGroupManager instance
        """
        super().__init__()
        
        self.device_mesh = device_mesh
        self.rank = rank
        self.world_size = pp_size
        self.group = pp_group
        self.is_first_stage = (self.rank == 0)
        self.is_last_stage = (self.rank == self.world_size-1)
        self.tensor_shapes = None
        # Get model depth (number of transformer blocks)
        self.depth = len(model.blocks)
        
        # Determine which layers belong to this stage
        self.layer_distribution = self.distribute_layers(self.depth)
        
        # Build local module for this stage
        self.local_module = self._build_local_module(model)
        
        # Move to correct device
        self.device = device
        self.local_module = self.local_module.to(self.device)

        print(f"[PipelineWrapper Rank {self.rank}] Initialized with blocks {self.layer_distribution}")
    
    def distribute_layers(self, num_layers):
        """
        Distribute transformer blocks across GPUs as evenly as possible.
        
        Args:
            num_layers: Total number of transformer blocks
        
        Returns:
            List of block indices for this pipeline stage
        """
        # Calculate blocks per GPU
        layers_per_gpu = [
            num_layers // self.world_size + (1 if i < num_layers % self.world_size else 0)
            for i in range(self.world_size)
        ]
        
        # Calculate starting block for this GPU
        start_layer = sum(layers_per_gpu[:self.rank])
        end_layer = start_layer + layers_per_gpu[self.rank]
        
        return list(range(start_layer, end_layer))
    
    def _build_local_module(self, model):
        """
        Build the local module for this pipeline stage.
        
        Args:
            model: The complete model
        
        Returns:
            nn.Sequential containing the layers for this stage
        """
        modules = []
        
        # First stage: add embedding
        if self.is_first_stage:
            modules.append(model.embedding)
            self.tensor_shapes = model.embedding.batch_size, model.embedding.in_channels, model.embedding.img_size, model.embedding.img_size


        # All stages: add assigned transformer blocks
        for block_idx in self.layer_distribution:
            modules.append(model.blocks[block_idx])
            self.tensor_shapes = model.blocks[block_idx].hidden_dim, model.blocks[block_idx].n_heads
        
        # Last stage: add classification head
        if self.is_last_stage:
            modules.append(model.classification_head)
        
        return nn.Sequential(*modules)

    def _get_tensor_shapes(self, batch_size):
        "For each module we need what is the optimum tensor shape that each pp stage should have as an input"
        return self.tensor_shapes
            


    def forward(self, x):
        """
        Forward pass through this pipeline stage.
        
        Args:
            x: Input tensor (images for first stage, hidden states for others)
        
        Returns:
            Output tensor for next stage or final predictions
        """
        return self.local_module(x)
    
    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """
        Backward pass for this pipeline stage.
        
        Args:
            input_tensor: Input to forward pass (to compute input gradients)
            output_tensor: Output from forward pass
            output_tensor_grad: Gradient from next stage
        
        Returns:
            Gradient with respect to input tensor
        """
        # Retain gradient for input tensor if it exists
        if input_tensor is not None:
            input_tensor.retain_grad()
        
        # If no gradient provided (last stage), create ones tensor
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(
                output_tensor, 
                memory_format=torch.preserve_format
            )
        
        # Perform backward pass
        torch.autograd.backward(
            output_tensor,
            grad_tensors=output_tensor_grad,
            retain_graph=False,
            create_graph=False
        )
        
        # Return input gradient to send to previous stage
        return input_tensor.grad if input_tensor is not None else None
