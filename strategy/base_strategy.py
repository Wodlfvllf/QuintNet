"""
Base Strategy Abstraction

This module defines the abstract base class for all parallelism strategies.
It ensures that every strategy, regardless of its complexity, presents a
single, consistent method (`.apply()`) to the user.

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

The `BaseStrategy` is not used directly, but all concrete strategies inherit
from it. This guarantees they all have the `.apply()` method, which is the
core of the strategy pattern.

.. code-block:: python

    # In strategy/dp_strategy.py
    class DataParallelStrategy(BaseStrategy):
        def apply(self, model):
            # ... logic to apply data parallelism
            return parallel_model

    # In the main training script
    strategy = get_strategy("dp", pg_manager, config)
    parallel_model = strategy.apply(model) # This is the unified API

===============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import os
from ..core import ProcessGroupManager

class BaseStrategy(ABC):
    """
    Abstract base class for all parallelism strategies.

    This class defines the common interface that all strategy classes must
    adhere to. It ensures that each strategy can be initialized with a process
    group manager and a configuration, and provides a single `apply` method
    to parallelize a model.
    """
    def __init__(self, pg_manager: ProcessGroupManager, config: Dict[str, Any]):
        """
        Initializes the BaseStrategy.

        Args:
            pg_manager (ProcessGroupManager): The process group manager, which
                provides information about the distributed setup.
            config (dict): The configuration dictionary.
        """
        self.pg_manager = pg_manager
        self.config = config
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """
        Determines the correct device for the current process based on the
        LOCAL_RANK environment variable.

        Returns:
            torch.device: The CUDA device for the current rank.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Abstract method to apply the specific parallelism strategy to a model.

        This method must be implemented by all concrete strategy subclasses.

        Args:
            model (nn.Module): The base `torch.nn.Module` to be parallelized.

        Returns:
            nn.Module: The parallelized model.
        """
        pass