"""
Tests for Data Parallelism (CustomDDP).

This module contains unit tests for the `CustomDDP` implementation, ensuring
that it correctly synchronizes gradients across multiple processes.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Distributed Data Parallel (DDP) is a core strategy for scaling training.
The `CustomDDP` implementation aims to provide a modular and efficient way
to achieve this. This test verifies the most critical aspect of DDP:
-   **Gradient Synchronization**: After each backward pass, the gradients
    computed on each model replica must be averaged across all participating
    processes. This test ensures that `CustomDDP` performs this averaging
    correctly, resulting in identical gradients on all ranks, which also
    match a manually computed averaged gradient.

The test works by:
1.  Initializing two identical models: one to be wrapped by `CustomDDP` and
    one to serve as a reference for manual gradient calculation.
2.  Performing a forward and backward pass on the `CustomDDP` wrapped model
    with different data on each rank.
3.  Manually simulating the DDP behavior on the reference model: collecting
    all data, running a single forward/backward pass, and then averaging
    the gradients.
4.  Comparing the gradients from the `CustomDDP` model with the manually
    averaged gradients from the reference model.

This test is marked with `@pytest.mark.world_size(N)` to indicate the
required number of processes for execution, and it relies on the `distributed_env`
and `device` fixtures from `conftest.py` for setup.

===============================================================================
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from ..parallelism.data_parallel import CustomDDP
from ..core.process_groups import ProcessGroupManager # Used for consistency, though not strictly by CustomDDP itself

@pytest.mark.world_size(2)
def test_ddp_gradient_synchronization(distributed_env, device):
    """
    Tests that `CustomDDP` correctly synchronizes gradients across ranks.

    This test verifies that after a backward pass, the gradients of the
    `CustomDDP` wrapped model are identical across all ranks and match
    the gradients obtained from a manually averaged reference model.

    This test requires a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 2:
        pytest.skip("This test requires a world size of at least 2.")

    # Set seed for reproducible weights and data across all ranks
    # Adding rank to seed ensures different data on each rank for DDP,
    # but initial model weights should be the same.
    torch.manual_seed(42 + rank)
    
    # Create two identical models: one for DDP, one for reference gradient calculation
    model_ddp = nn.Linear(10, 5, bias=False).to(device)
    model_ref = nn.Linear(10, 5, bias=False).to(device)
    
    # Ensure both models start with the exact same weights by broadcasting from rank 0
    with torch.no_grad():
        dist.broadcast(model_ddp.weight, src=0)
        model_ref.weight.copy_(model_ddp.weight)

    # Wrap one model with CustomDDP
    # The ProcessGroupManager is created for consistency with other tests,
    # but CustomDDP itself can use the default process group if not specified.
    pg_manager = ProcessGroupManager(mesh_dim=(world_size, 1), mesh_name=('dp', 'tp'))
    ddp_model = CustomDDP(model_ddp, distributed_config=pg_manager.config) # Pass distributed config if needed

    # Create different input data for each rank
    input_tensor = torch.randn(4, 10).to(device)
    
    # --- DDP Model Forward/Backward Pass ---
    # This will trigger CustomDDP's gradient bucketing and all-reduce.
    output_ddp = ddp_model(input_tensor)
    loss_ddp = output_ddp.sum()
    loss_ddp.backward()

    # --- Reference Model Gradient Calculation ---
    # To get the equivalent gradient for comparison, we must:
    # 1. Gather all input data from all ranks.
    # 2. Run a forward/backward pass on the *full* dataset (concatenated inputs)
    #    using the reference model.
    # 3. Manually average the gradients by dividing by the world size.
    
    # Gather all inputs from all ranks to form a global batch
    all_inputs = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_tensor)
    
    # Process all data chunks on the reference model
    model_ref.zero_grad()
    for i in range(world_size):
        output_ref = model_ref(all_inputs[i])
        loss_ref = output_ref.sum()
        # Accumulate gradients by dividing loss by world_size.
        # This simulates the averaging effect of DDP's all-reduce.
        (loss_ref / world_size).backward()

    # At this point, `model_ref.weight.grad` contains the manually averaged gradient.
    
    # --- Compare the gradients ---
    # The gradients on the DDP model's parameters should now be very close to
    # the manually averaged gradients on the reference model.
    assert torch.allclose(ddp_model.model.weight.grad, model_ref.weight.grad, atol=1e-6), \
        f"Gradients do not match between CustomDDP and reference model on rank {rank}."

    # As an extra check, the gradients on both ranks of the DDP model should be identical.
    # We gather the DDP model's gradients from all ranks and compare them.
    grad_tensor = ddp_model.model.weight.grad.clone()
    grad_list = [torch.empty_like(grad_tensor) for _ in range(world_size)]
    dist.all_gather(grad_list, grad_tensor)
    
    # Assert that the gradients on rank 0 and rank 1 are identical
    assert torch.allclose(grad_list[0], grad_list[1], atol=1e-6), "Gradients on DDP models across ranks are not identical."
