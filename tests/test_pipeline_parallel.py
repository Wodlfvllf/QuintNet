"""
Tests for Pipeline Parallelism.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper
from QuintNet.utils.model import Model, TransformerBlock

@pytest.mark.world_size(2)
def test_layer_distribution(distributed_env, device):
    """
    Tests that layers are distributed correctly across two pipeline stages.
    This test must be run with a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    # Create a model with a known number of layers (e.g., 8 transformer blocks)
    depth = 8
    model = Model(depth=depth)

    # Create a ProcessGroupManager for the test
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

    # With 8 layers and 2 GPUs, each should get 4 transformer blocks.
    expected_blocks_per_stage = depth // world_size
    
    # Count the number of TransformerBlock instances in the local module
    num_local_blocks = sum(1 for m in pp_wrapper.local_module if isinstance(m, TransformerBlock))
    
    assert num_local_blocks == expected_blocks_per_stage, f"Rank {rank} should have {expected_blocks_per_stage} blocks, but has {num_local_blocks}."

    # Additionally, check for the embedding/classification head
    if rank == 0:
        # First stage should have the embedding layer
        assert any(isinstance(m, nn.Module) and "embedding" in m.__class__.__name__.lower() for m in pp_wrapper.local_module)
        assert not any(isinstance(m, nn.Module) and "classification" in m.__class__.__name__.lower() for m in pp_wrapper.local_module)
    else:
        # Last stage should have the classification head
        assert not any(isinstance(m, nn.Module) and "embedding" in m.__class__.__name__.lower() for m in pp_wrapper.local_module)
        assert any(isinstance(m, nn.Module) and "classification" in m.__class__.__name__.lower() for m in pp_wrapper.local_module)


@pytest.mark.world_size(2)
def test_pipeline_forward_pass(distributed_env, device):
    """
    Tests the forward pass of a 2-stage pipeline parallel model.
    This test must be run with a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    torch.manual_seed(42)
    
    # Model parameters
    img_size = 28
    patch_size = 4
    hidden_dim = 64
    depth = 4 # 2 blocks per stage
    batch_size = 2
    num_classes = 10

    # Create a model
    model = Model(img_size=img_size, patch_size=patch_size, hidden_dim=hidden_dim, depth=depth, n_heads=4)

    # Create a ProcessGroupManager for the test
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

    # Create a sample input tensor for the first stage
    if rank == 0:
        input_tensor = torch.randn(batch_size, 1, img_size, img_size).to(device)
        
        # Run the forward pass on the first stage
        output_tensor = pp_wrapper(input_tensor)
        
        # Send the intermediate activation to the next stage
        dist.send(output_tensor, dst=1, group=pp_group)
    else: # rank == 1
        # Determine the shape of the intermediate tensor
        num_patches = (img_size // patch_size) ** 2
        # Shape is (batch_size, num_patches + 1 for CLS token, hidden_dim)
        intermediate_shape = (batch_size, num_patches + 1, hidden_dim)
        received_tensor = torch.empty(intermediate_shape, device=device)
        
        # Receive the activation from the previous stage
        dist.recv(received_tensor, src=0, group=pp_group)
        
        # Run the forward pass on the second stage
        final_output = pp_wrapper(received_tensor)
        
        # The final output on the last stage should have the shape (batch_size, num_classes)
        assert final_output.shape == (batch_size, num_classes), f"Expected output shape {(batch_size, num_classes)}, but got {final_output.shape}."
        
        # Check that the output is a valid tensor
        assert not torch.isnan(final_output).any(), "Output contains NaNs."
        assert not torch.isinf(final_output).any(), "Output contains Infs."