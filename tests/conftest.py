"""
Pytest Configuration and Fixtures for Distributed Testing.

This module provides essential pytest fixtures to set up and tear down
a distributed environment for running tests. It ensures that tests can
be executed correctly in both single-process (local) and multi-process
(distributed) contexts.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Testing distributed systems is inherently complex. This `conftest.py`
aims to simplify this by:

-   **`distributed_env` fixture**: Automatically initializes and destroys
    the `torch.distributed` process group for each test function that
    requires it. It intelligently handles environment variables set by
    launchers like `torchrun` and provides sensible defaults for local
    execution.
-   **`device` fixture**: Provides the correct `torch.device` for the
    current process, respecting `LOCAL_RANK` in a distributed setup.
-   **`dummy_model` fixture**: Offers a simple `nn.Module` for quick
    testing of parallelism components without needing a complex model.

These fixtures allow test writers to focus on the logic of their tests
rather than the boilerplate of distributed setup.

===============================================================================
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from typing import Generator

def is_distributed() -> bool:
    """
    Checks if the `torch.distributed` environment is currently initialized.

    Returns:
        bool: True if distributed is initialized, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()

@pytest.fixture(scope="function")
def distributed_env(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Initializes and tears down the distributed environment for a single test function.

    This fixture is designed to be used with tests launched via `torchrun`
    or similar distributed runners. It can also provide a basic distributed
    setup for local testing if environment variables are not fully configured.

    Usage:
        @pytest.mark.world_size(2) # Mark test to run with 2 processes
        def test_my_distributed_function(distributed_env):
            # ... test logic ...

    Args:
        request (pytest.FixtureRequest): The pytest request object, used to
            access test markers.

    Yields:
        None: The fixture yields control to the test function.
    """
    # Get world size from the test marker, default to 1 if no marker is present.
    # Example: @pytest.mark.world_size(4)
    marker = request.node.get_closest_marker("world_size")
    expected_world_size = marker.args[0] if marker else 1

    # Set default environment variables for local testing if not already set.
    # These are typically set by `torchrun`.
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355') # Use a unique port for tests
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', str(expected_world_size))
    os.environ.setdefault('LOCAL_RANK', '0') # For device selection

    current_rank = int(os.environ.get('RANK', '0'))
    current_world_size = int(os.environ.get('WORLD_SIZE', '1'))

    # Skip the test if the environment's world size doesn't match the test's expectation.
    # This is important when running a subset of tests or if the launcher is misconfigured.
    if current_world_size != expected_world_size:
        pytest.skip(f"Test requires world_size={expected_world_size}, but environment is WORLD_SIZE={current_world_size}")

    # Initialize the distributed process group if not already initialized.
    if not dist.is_initialized():
        if expected_world_size > 1:
            # Use NCCL backend for multi-GPU/multi-node tests
            dist.init_process_group(backend="nccl", rank=current_rank, world_size=current_world_size, timeout=torch.distributed.default_pg_timeout)
        else:
            # Use Gloo backend for single-process (CPU or single-GPU) tests
            dist.init_process_group(backend="gloo", rank=0, world_size=1, timeout=torch.distributed.default_pg_timeout)

    yield # Yield control to the test function

    # Teardown: Destroy the process group after the test function completes.
    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.fixture
def device() -> torch.device:
    """
    Fixture for device selection that is aware of the distributed environment.

    It returns a CUDA device if available and `LOCAL_RANK` is set, otherwise CPU.

    Returns:
        torch.device: The appropriate `torch.device` for the current process.
    """
    if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')

@pytest.fixture
def dummy_model() -> nn.Module:
    """
    Fixture for creating a simple dummy `nn.Module` for testing purposes.

    This model can be used to test parallelism wrappers or other components
    without needing a complex, full-fledged model.

    Returns:
        nn.Module: A simple sequential model.
    """
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
