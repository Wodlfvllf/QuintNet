"""
Tests for Tensor Parallelism Layers.

This module contains unit tests for the `ColumnParallelLinear` and
`RowParallelLinear` layers, which are fundamental components of tensor
parallelism in QuintNet.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Tensor parallelism involves sharding the weights of individual layers across
multiple devices. These tests verify that the custom parallel layers produce
numerically identical results to their standard `nn.Linear` counterparts
when run in a distributed environment.

-   **`test_column_parallel_linear`**: Verifies that a `ColumnParallelLinear`
    layer, which shards the output features of a linear layer, correctly
    computes its output and gathers it across ranks to match a standard
    `nn.Linear` layer.
-   **`test_row_parallel_linear`**: Verifies that a `RowParallelLinear` layer,
    which shards the input features of a linear layer, correctly computes
    its output and performs an all-reduce across ranks to match a standard
    `nn.Linear` layer.

These tests are marked with `@pytest.mark.world_size(N)` to indicate the
required number of processes for execution, and they rely on the `distributed_env`
and `device` fixtures from `conftest.py` for setup.

===============================================================================
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
    Tests the `ColumnParallelLinear` layer.

    It compares the output of a `ColumnParallelLinear` layer (with `gather_output=True`)
    against a standard `nn.Linear` layer. The weights of the standard layer are
    sharded and distributed to the parallel layer.

    This test requires a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    in_features = 4
    out_features = 6
    
    # Use a fixed seed for reproducibility across ranks
    torch.manual_seed(42)
    
    # Create a standard linear layer as the ground truth (only on rank 0 for initialization)
    base_linear = nn.Linear(in_features, out_features, bias=False).to(device)
    # Broadcast the initial weights to ensure all ranks have the same starting point
    dist.broadcast(base_linear.weight, src=0)

    # Create a ProcessGroupManager for the test, defining a 1D TP mesh
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'tp'))
    tp_group = pg_manager.get_group('tp')

    # Shard the weight for the parallel layer (split along the output dimension, dim=0)
    weight_shard = base_linear.weight.chunk(world_size, dim=0)[rank].detach().clone()
    
    # Create the ColumnParallelLinear layer
    col_parallel_linear = ColumnParallelLinear(
        local_device=device,
        tp_group=tp_group,
        in_features=in_features,
        out_features_per_rank=out_features // world_size,
        weight_slice=weight_shard,
        bias_slice=None, # For simplicity, test without bias
        gather_output=True # Ensure output is gathered to compare with base_linear
    )

    # Create a sample input tensor and ensure all ranks have the same input
    input_tensor = torch.randn(2, in_features).to(device)
    dist.broadcast(input_tensor, src=0) # Broadcast input to all ranks

    # Forward pass for both layers
    base_output = base_linear(input_tensor)
    parallel_output = col_parallel_linear(input_tensor)

    # The outputs should be numerically identical
    assert torch.allclose(base_output, parallel_output, atol=1e-6), "ColumnParallelLinear output does not match base layer."


@pytest.mark.world_size(2)
def test_row_parallel_linear(distributed_env, device):
    """
    Tests the `RowParallelLinear` layer.

    It compares the output of a `RowParallelLinear` layer against a standard
    `nn.Linear` layer. The weights of the standard layer are sharded and
    distributed to the parallel layer.

    This test requires a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 2:
        pytest.skip("This test requires a world size of 2.")

    in_features = 4
    out_features = 6

    # Use a fixed seed for reproducibility across ranks
    torch.manual_seed(42)
    
    # Create a standard linear layer as the ground truth (only on rank 0 for initialization)
    base_linear = nn.Linear(in_features, out_features, bias=False).to(device)
    # Broadcast the initial weights to ensure all ranks have the same starting point
    dist.broadcast(base_linear.weight, src=0)

    # Create a ProcessGroupManager for the test, defining a 1D TP mesh
    pg_manager = ProcessGroupManager(mesh_dim=(1, 2), mesh_name=('dp', 'tp'))
    tp_group = pg_manager.get_group('tp')

    # Shard the weight for the parallel layer (split along the input dimension, dim=1)
    weight_shard = base_linear.weight.chunk(world_size, dim=1)[rank].detach().clone()

    # Create the RowParallelLinear layer
    row_parallel_linear = RowParallelLinear(
        local_device=device,
        tp_group=tp_group,
        in_features_per_rank=in_features // world_size,
        out_features=out_features,
        weight_slice=weight_shard,
        input_is_parallel=False # Input is replicated, not pre-sharded, so layer will slice it
    )

    # Create a sample input tensor and ensure all ranks have the same input
    input_tensor = torch.randn(2, in_features).to(device)
    dist.broadcast(input_tensor, src=0) # Broadcast input to all ranks

    # Forward pass for both layers
    base_output = base_linear(input_tensor)
    parallel_output = row_parallel_linear(input_tensor)

    # The outputs should be numerically identical after the all-reduce operation in RowParallelLinear
    assert torch.allclose(base_output, parallel_output, atol=1e-6), "RowParallelLinear output does not match base layer."
