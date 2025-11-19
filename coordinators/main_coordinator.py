"""
Base Coordinator Abstraction

This module defines the abstract base class for all parallelism coordinators.
It ensures that every coordinator, which handles the logic for applying a
parallelism strategy, presents a single, consistent method (`.parallelize()`).

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

The `BaseCoordinator` is not used directly, but all concrete coordinators
inherit from it. This guarantees they all have the `.parallelize()` method,
which is called by the strategy's `.apply()` method.

.. code-block:: python

    # In coordinators/dp_coordinator.py
    class DataParallelCoordinator(BaseCoordinator):
        def parallelize(self):
            # ... logic to apply data parallelism
            return parallel_model

    # In strategy/dp_strategy.py
    class DataParallelStrategy(BaseStrategy):
        def apply(self, model):
            coordinator = DataParallelCoordinator(model, ...)
            return coordinator.parallelize() # This is the unified API

===============================================================================
"""

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseCoordinator(ABC):
    """
    Abstract base class for all parallelism coordinators.

    This class defines the common interface that all coordinator classes must
    adhere to. It ensures that each coordinator can be initialized with a model
    and provides a single `parallelize` method to apply its specific logic.
    """
    def __init__(self, model: nn.Module, **kwargs):
        """
        Initializes the BaseCoordinator.

        Args:
            model (nn.Module): The base `torch.nn.Module` to be parallelized.
            **kwargs: A dictionary to absorb any additional parameters that
                a specific coordinator might need, passed down from the
                strategy. This allows for a flexible constructor.
        """
        self.model = model
        # Absorb any other parameters that might be passed from the strategy
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def parallelize(self) -> nn.Module:
        """
        Abstract method to apply the coordinator's parallelism logic.

        This method must be implemented by all concrete coordinator subclasses.
        It contains the orchestration logic for applying one or more
        parallelism techniques.

        Returns:
            nn.Module: The parallelized model.
        """
        pass