"""
Pytest Configuration and Fixtures for Distributed Testing.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import os

def is_distributed():
    """Check if the distributed environment is initialized."""
    return dist.is_available() and dist.is_initialized()

@pytest.fixture(scope="function")
def distributed_env(request):
    """
    Initializes the distributed environment for a single test function.
    This fixture assumes that tests are launched with a distributed runner
    like `torchrun`. It sets default values for local testing if the
    environment variables are not present.
    """
    # Get world size from the test marker, default to 1 if not present
    marker = request.node.get_closest_marker("world_size")
    world_size = marker.args[0] if marker else 1

    # Set default environment variables for local testing
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    rank = int(os.environ.get('RANK', '0'))
    world_size_env = int(os.environ.get('WORLD_SIZE', world_size))

    # Ensure the environment's world size matches the test's expectation
    if 'WORLD_SIZE' in os.environ and world_size != world_size_env:
        pytest.skip(f"Test requires world_size={world_size}, but environment is WORLD_SIZE={world_size_env}")

    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        else:
            # Use gloo backend for single-process (CPU) tests
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    yield

    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.fixture
def device():
    """Fixture for device selection that is aware of the distributed environment."""
    if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')

@pytest.fixture
def dummy_model():
    """Fixture for creating a simple dummy model for testing purposes."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )