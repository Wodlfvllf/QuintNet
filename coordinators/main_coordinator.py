from abc import ABC, abstractmethod
import torch.nn as nn

class BaseCoordinator(ABC):
    """
    Abstract base class for all parallelism coordinators.
    It defines a common interface for applying a parallelism strategy.
    """
    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        # Absorb any other parameters that might be passed
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def parallelize(self) -> nn.Module:
        """
        This method should be implemented by all concrete coordinators
        to apply their specific parallelism logic to the model.
        """
        pass
