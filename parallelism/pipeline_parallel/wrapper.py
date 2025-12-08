"""
Pipeline Parallel Wrapper

This module contains the `PipelineParallelWrapper`, which is responsible for
splitting a standard `nn.Module` into multiple stages for pipeline parallelism.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Pipeline parallelism involves splitting the layers of a model across multiple
devices (GPUs). Each device holds a "stage" of the model. During a training
step, data flows through these stages sequentially.

Consider a 4-layer model on 2 GPUs:
- GPU 0 (Stage 0) holds Layer 1 and Layer 2.
- GPU 1 (Stage 1) holds Layer 3 and Layer 4.

The `PipelineParallelWrapper` automates this process. When initialized on a
specific rank, it determines which layers belong to that rank's stage and
constructs a local `nn.Sequential` module containing only those layers.

It also handles the custom `backward` pass required for pipeline parallelism,
which is not a standard `nn.Module` method but is called by the `PipelineTrainer`.

===============================================================================
"""

import torch
import torch.nn as nn
import torch.distributed as dist

class PipelineParallelWrapper(nn.Module):
    """
    Wraps a model (specifically, the custom ViT `Model`) for pipeline parallelism.

    This class automatically distributes the transformer blocks of the model
    across multiple devices, creating a pipeline stage on each device.
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
        Initializes the PipelineParallelWrapper.

        Args:
            model (nn.Module): The complete model to be split into stages.
                This wrapper is specifically designed for the `Model` class
                in `QuintNet.utils.model`.
            device_mesh: The device mesh object, used to get rank information.
            rank (int): The rank of the current process within the pipeline
                parallel group.
            pp_group (dist.ProcessGroup): The process group for pipeline
                parallel communication.
            pp_size (int): The total number of stages in the pipeline.
            device (torch.device): The CUDA device for the current process.
        """
        super().__init__()
        
        self.device_mesh = device_mesh
        self.rank = rank
        self.world_size = pp_size
        self.group = pp_group
        self.is_first_stage = (self.rank == 0)
        self.is_last_stage = (self.rank == self.world_size - 1)
        self.tensor_shapes = None # Shape of the input tensor for this stage
        
        # The wrapper assumes the model has a `blocks` attribute which is a list of layers.
        self.depth = len(model.blocks)
        
        # Determine which layers (transformer blocks) belong to this stage.
        self.layer_distribution = self._distribute_layers(self.depth)
        
        # Build the local module for this stage from the full model.
        self.local_module = self._build_local_module(model)
        
        # Move the local module to the correct device.
        self.device = device
        self.local_module = self.local_module.to(self.device)

        # Print a message on the first DP rank of each stage for clarity.
        if self.device_mesh.get_coordinates_tensor_search(dist.get_rank())[0] == 0:
            pass
    
    def _distribute_layers(self, num_layers: int) -> list:
        """
        Calculates which layer indices belong to the current pipeline stage.

        This method distributes layers as evenly as possible across all stages.

        Args:
            num_layers (int): The total number of layers to distribute.

        Returns:
            list: A list of layer indices assigned to the current rank.
        """
        # Calculate the number of layers for each GPU/stage.
        # The modulo operator ensures that any remainder layers are distributed
        # one by one to the first few stages.
        layers_per_gpu = [
            num_layers // self.world_size + (1 if i < num_layers % self.world_size else 0)
            for i in range(self.world_size)
        ]
        
        # Calculate the start and end indices for the current rank's layers.
        start_layer = sum(layers_per_gpu[:self.rank])
        end_layer = start_layer + layers_per_gpu[self.rank]
        
        return list(range(start_layer, end_layer))
    
    def _build_local_module(self, model: nn.Module) -> nn.Sequential:
        """
        Constructs the `nn.Sequential` module for the current pipeline stage.

        This method assembles the correct layers (embedding, transformer blocks,
        classification head) based on whether it is the first, middle, or last
        stage in the pipeline.

        Args:
            model (nn.Module): The complete, original model.

        Returns:
            nn.Sequential: The module representing the local pipeline stage.
        """
        modules = []
        
        # The first stage is unique: it includes the model's embedding layer.
        if self.is_first_stage:
            # For ALL stages (including the first), we need to communicate the shape
            # of the embedding/activation tensor that flows through the pipeline.
            # This is always (B, Seq_len, Hidden_dim).
            
            # We determine Seq_len dynamically from the positional embeddings.
            # This works for both 2D and 3D models (or any model with pos_embed).
            # pos_embed shape is (1, seq_len, hidden_dim)
            seq_len = model.embedding.pos_embed.shape[1]
            
            self.tensor_shapes = (
                seq_len,
                model.blocks[0].hidden_dim # Assumes hidden_dim is consistent
            )

            modules.append(model.embedding)
        else:
            # Subsequent stages receive activations from the previous stage.
            # We calculate the shape of these activations for communication.
            
            # Same logic as above: derive from pos_embed for consistency and 3D support.
            seq_len = model.embedding.pos_embed.shape[1]
            
            self.tensor_shapes = (
                seq_len,
                model.blocks[0].hidden_dim 
            )

        # All stages include their assigned transformer blocks.
        for block_idx in self.layer_distribution:
            modules.append(model.blocks[block_idx])
        
        # The last stage is unique: it includes the final classification head.
        if self.is_last_stage:
            modules.append(model.classification_head)
        
        return nn.Sequential(*modules)

    def get_tensor_shapes(self, batch_size: int) -> tuple:
        """
        Gets the expected shape of the input tensor for this stage, including batch size.

        Args:
            batch_size (int): The current batch size.

        Returns:
            tuple: The full shape of the input tensor for this stage.
        """
        return (batch_size, *self.tensor_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for only the layers in this pipeline stage.

        Args:
            x (torch.Tensor): The input tensor. For the first stage, this is the
                batch of images. For subsequent stages, it is the activation
                tensor from the previous stage.

        Returns:
            torch.Tensor: The output activation tensor to be passed to the next stage.
        """
        res = self.local_module(x)
        return res
    
    def backward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, output_tensor_grad: torch.Tensor) -> torch.Tensor:
        """
        Performs the backward pass for this pipeline stage.

        This is a custom method called by the `PipelineTrainer`, not by `torch.autograd`.

        Args:
            input_tensor (torch.Tensor): The input tensor that was saved from the forward pass.
            output_tensor (torch.Tensor): The output tensor that was saved from the forward pass.
            output_tensor_grad (torch.Tensor): The gradient of the loss with respect
                to the output of this stage, received from the next stage.

        Returns:
            torch.Tensor: The gradient of the loss with respect to the input of this
                stage, which will be sent to the previous stage.
        """
        # We need to retain the grad on the input tensor if it's not a leaf,
        # so we can compute the gradient w.r.t. it.
        if input_tensor is not None:
            input_tensor.retain_grad()
        
        # For the last stage, the `output_tensor_grad` is None, as it's the start
        # of the backward pass. We initialize it with ones.
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(
                output_tensor, 
                memory_format=torch.preserve_format
            )
        
        # Execute the backward pass for the local module.
        torch.autograd.backward(
            tensors=(output_tensor,),
            grad_tensors=(output_tensor_grad,),
        )
        
        # Return the gradient of the input tensor, which will be passed to the previous stage.
        return input_tensor.grad if input_tensor is not None else None