"""
Tests for Pipeline Parallelism.

This module contains unit tests for the `PipelineParallelWrapper` class,
which is responsible for splitting a model into stages for pipeline parallelism.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Pipeline parallelism distributes layers of a model across multiple devices.
These tests verify the correct functionality of the `PipelineParallelWrapper`:
-   **Layer Distribution**: Ensures that the model's layers (specifically
    transformer blocks, embedding, and classification head) are correctly
    assigned to their respective pipeline stages.
-   **Forward Pass Communication**: Verifies that intermediate activations
    are correctly passed between pipeline stages during the forward pass,
    leading to a valid final output on the last stage.

These tests are marked with `@pytest.mark.world_size(N)` to indicate the
required number of processes for execution, and they rely on the `distributed_env`
and `device` fixtures from `conftest.py` for setup.

===============================================================================
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from ..core.process_groups import ProcessGroupManager
from ..parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper
from ..utils.model import Model, TransformerBlock, ViTEmbedding, ClassificationHead

@pytest.mark.world_size(2)
def test_layer_distribution(distributed_env, device):
    """
    Tests that model layers (embedding, transformer blocks, classification head)
    are correctly distributed across two pipeline stages.

    This test requires a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    # Create a model with a known number of layers (e.g., 8 transformer blocks)
    depth = 8
    model = Model(depth=depth)

    # Create a ProcessGroupManager for the test, defining a 1D PP mesh
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'pp'))
    pp_group = pg_manager.get_group('pp')

    # Wrap the model for pipeline parallelism
    pp_wrapper = PipelineParallelWrapper(
        model,
        pg_manager.device_mesh,
        rank,
        pp_group,
        world_size,
        device
    )

    # With 8 transformer blocks and 2 pipeline stages, each stage should get 4 blocks.
    expected_blocks_per_stage = depth // world_size
    
    # Count the number of TransformerBlock instances in the local module
    num_local_blocks = sum(1 for m in pp_wrapper.local_module if isinstance(m, TransformerBlock))
    
    assert num_local_blocks == expected_blocks_per_stage, \
        f"Rank {rank} should have {expected_blocks_per_stage} blocks, but has {num_local_blocks}."

    # Additionally, check for the presence of embedding/classification head
    if rank == 0:
        # The first stage should contain the embedding layer
        assert any(isinstance(m, ViTEmbedding) for m in pp_wrapper.local_module), "First stage missing embedding."
        # The first stage should NOT contain the classification head
        assert not any(isinstance(m, ClassificationHead) for m in pp_wrapper.local_module), "First stage should not have classification head."
    else: # rank == 1 (last stage)
        # The last stage should NOT contain the embedding layer
        assert not any(isinstance(m, ViTEmbedding) for m in pp_wrapper.local_module), "Last stage should not have embedding."
        # The last stage should contain the classification head
        assert any(isinstance(m, ClassificationHead) for m in pp_wrapper.local_module), "Last stage missing classification head."


@pytest.mark.world_size(2)
def test_pipeline_forward_pass(distributed_env, device):
    """
    Tests the forward pass of a 2-stage pipeline parallel model, including
    inter-stage communication.

    This test requires a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    torch.manual_seed(42) # For reproducibility
    
    # Model parameters
    img_size = 28
    patch_size = 4
    hidden_dim = 64
    depth = 4 # 2 blocks per stage
    batch_size = 2
    num_classes = 10

    # Create a model instance
    model = Model(img_size=img_size, patch_size=patch_size, hidden_dim=hidden_dim, depth=depth, n_heads=4, num_classes=num_classes)

    # Create a ProcessGroupManager for the test, defining a 1D PP mesh
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'pp'))
    pp_group = pg_manager.get_group('pp')

    # Wrap the model for pipeline parallelism
    pp_wrapper = PipelineParallelWrapper(
        model,
        pg_manager.device_mesh,
        rank,
        pp_group,
        world_size,
        device
    )

    # --- Rank 0 (First Stage) ---
    if rank == 0:
        # Create a sample input tensor (images)
        input_tensor = torch.randn(batch_size, 1, img_size, img_size).to(device)
        
        # Run the forward pass on the first stage's local module
        output_tensor = pp_wrapper(input_tensor)
        
        # Send the intermediate activation to the next stage (rank 1)
        dist.send(output_tensor, dst=1, group=pp_group)
        
        # Assert that the output tensor has the expected intermediate shape
        num_patches = (img_size // patch_size) ** 2
        expected_intermediate_shape = (batch_size, num_patches + 1, hidden_dim)
        assert output_tensor.shape == expected_intermediate_shape, \
            f"Rank {rank}: Expected intermediate output shape {expected_intermediate_shape}, but got {output_tensor.shape}."

    # --- Rank 1 (Last Stage) ---
    else: # rank == 1
        # Determine the shape of the intermediate tensor to be received
        num_patches = (img_size // patch_size) ** 2
        intermediate_shape = (batch_size, num_patches + 1, hidden_dim)
        
        # Allocate a tensor to receive the activation from the previous stage
        received_tensor = torch.empty(intermediate_shape, device=device)
        
        # Receive the activation from the previous stage (rank 0)
        dist.recv(received_tensor, src=0, group=pp_group)
        
        # Run the forward pass on the second stage's local module
        final_output = pp_wrapper(received_tensor)
        
        # The final output on the last stage should have the shape (batch_size, num_classes)
        assert final_output.shape == (batch_size, num_classes), \
            f"Rank {rank}: Expected final output shape {(batch_size, num_classes)}, but got {final_output.shape}."
        
        # Check that the output is a valid tensor (no NaNs or Infs)
        assert not torch.isnan(final_output).any(), "Output contains NaNs."
        assert not torch.isinf(final_output).any(), "Output contains Infs."
