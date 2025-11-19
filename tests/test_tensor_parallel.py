"""
Tests for Tensor Parallelism Layers.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

@pytest.mark.world_size(2)
def test_column_parallel_linear(distributed_env, device):
    """
    Tests the ColumnParallelLinear layer by comparing its output to a standard nn.Linear layer.
    This test must be run with a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    in_features = 4
    out_features = 6
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Create a standard linear layer as the ground truth
    base_linear = nn.Linear(in_features, out_features, bias=False).to(device)

    # Create a ProcessGroupManager for the test
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'tp'))
    tp_group = pg_manager.get_group('tp')

    # Shard the weight for the parallel layer (split along the output dimension)
    weight_shard = base_linear.weight.chunk(world_size, dim=0)[rank].detach().clone()
    
    # Create the parallel layer
    col_parallel_linear = ColumnParallelLinear(
        local_device=device,
        tp_group=tp_group,
        in_features=in_features,
        out_features_per_rank=out_features // world_size,
        weight_slice=weight_shard,
        bias_slice=None,
        gather_output=True
    )

    # Create a sample input tensor and ensure all ranks have the same input
    input_tensor = torch.randn(2, in_features).to(device)
    dist.broadcast(input_tensor, src=0)

    # Forward pass for both layers
    base_output = base_linear(input_tensor)
    parallel_output = col_parallel_linear(input_tensor)

    # The outputs should be identical
    assert torch.allclose(base_output, parallel_output, atol=1e-6), "ColumnParallelLinear output does not match base layer."


@pytest.mark.world_size(2)
def test_row_parallel_linear(distributed_env, device):
    """
    Tests the RowParallelLinear layer by comparing its output to a standard nn.Linear layer.
    This test must be run with a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    in_features = 4
    out_features = 6

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Create a standard linear layer as the ground truth
    base_linear = nn.Linear(in_features, out_features, bias=False).to(device)

    # Create a ProcessGroupManager for the test
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'tp'))
    tp_group = pg_manager.get_group('tp')

    # Shard the weight for the parallel layer (split along the input dimension)
    weight_shard = base_linear.weight.chunk(world_size, dim=1)[rank].detach().clone()

    # Create the parallel layer
    row_parallel_linear = RowParallelLinear(
        local_device=device,
        tp_group=tp_group,
        in_features_per_rank=in_features // world_size,
        out_features=out_features,
        weight_slice=weight_shard,
        input_is_parallel=False # Input is replicated, not pre-sharded
    )

    # Create a sample input tensor and ensure all ranks have the same input
    input_tensor = torch.randn(2, in_features).to(device)
    dist.broadcast(input_tensor, src=0)

    # Forward pass for both layers
    base_output = base_linear(input_tensor)
    parallel_output = row_parallel_linear(input_tensor)

    # The outputs should be identical after the all-reduce operation in RowParallelLinear
    assert torch.allclose(base_output, parallel_output, atol=1e-6), "RowParallelLinear output does not match base layer."