"""
Tests for Data Parallelism (CustomDDP).
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.parallelism.data_parallel import CustomDDP
from QuintNet.core.process_groups import ProcessGroupManager

@pytest.mark.world_size(2)
def test_ddp_gradient_synchronization(distributed_env, device):
    """
    Tests that CustomDDP correctly synchronizes gradients across ranks.
    This test works by:
    1. Creating two identical models.
    2. Wrapping one with CustomDDP.
    3. Running a forward/backward pass on different data on each rank for the DDP model.
    4. For the reference model, manually gathering the data from all ranks, running a
       single forward/backward pass, and then averaging the gradients.
    5. The gradients of the DDP model's parameters should equal the manually averaged
       gradients of the reference model.
    
    This test must be run with a world size of 2.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 2:
        pytest.skip("This test requires a world size of at least 2.")

    # Set seed for reproducible weights and data
    torch.manual_seed(42 + rank)
    
    # Create two identical models, one for DDP and one for reference
    model_ddp = nn.Linear(10, 5, bias=False).to(device)
    model_ref = nn.Linear(10, 5, bias=False).to(device)
    
    # Ensure both models start with the exact same weights
    with torch.no_grad():
        dist.broadcast(model_ddp.weight, src=0)
        model_ref.weight.copy_(model_ddp.weight)

    # Wrap one model with CustomDDP
    # Note: CustomDDP doesn't use the pg_manager, but we create it for consistency
    pg_manager = ProcessGroupManager(mesh_dim=(world_size, 1), mesh_name=('dp', 'tp'))
    ddp_model = CustomDDP(model_ddp)

    # Create different input for each rank
    input_tensor = torch.randn(4, 10).to(device)
    
    # --- DDP Model Forward/Backward Pass ---
    output_ddp = ddp_model(input_tensor)
    loss_ddp = output_ddp.sum()
    loss_ddp.backward()

    # --- Reference Model Gradient Calculation ---
    # To get the equivalent gradient, we must run all data through the reference model
    # and then average the gradients.
    
    # Gather all inputs from all ranks
    all_inputs = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_tensor)
    
    # Process all data chunks on the reference model
    model_ref.zero_grad()
    for i in range(world_size):
        output_ref = model_ref(all_inputs[i])
        loss_ref = output_ref.sum()
        # Accumulate gradients by dividing loss by world_size
        (loss_ref / world_size).backward()

    # At this point, model_ref.weight.grad contains the averaged gradient.
    
    # --- Compare the gradients ---
    # The gradients on the DDP model should now be very close to the manually
    # averaged gradients on the reference model.
    assert torch.allclose(ddp_model.model.weight.grad, model_ref.weight.grad, atol=1e-6), \
        f"Gradients do not match between CustomDDP and reference model on rank {rank}."

    # As an extra check, the gradients on both ranks of the DDP model should be identical.
    grad_tensor = ddp_model.model.weight.grad.clone()
    grad_list = [torch.empty_like(grad_tensor) for _ in range(world_size)]
    dist.all_gather(grad_list, grad_tensor)
    
    assert torch.allclose(grad_list[0], grad_list[1], atol=1e-6), "Gradients on DDP models across ranks are not identical."