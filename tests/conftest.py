"""
Pytest Configuration and Fixtures.

TODO: Add common test fixtures
"""

import pytest
import torch
import torch.distributed as dist


@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_model():
    """Fixture for creating a dummy model."""
    # TODO: Implement
    pass
