from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import os
from QuintNet.core.process_groups import ProcessGroupManager

class BaseStrategy(ABC):
    """
    Abstract base class for all parallelism strategies.
    """
    def __init__(self, pg_manager: ProcessGroupManager, config: Dict[str, Any]):
        self.pg_manager = pg_manager
        self.config = config
        self.device = self._get_device()

    def _get_device(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply the parallelism strategy to the model.
        """
        pass
