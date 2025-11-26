import torch.nn as nn
from .base_strategy import BaseStrategy

class DataTensorParallelStrategy(BaseStrategy):
    """
    Data and Tensor Parallelism Strategy (DP+TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from ..coordinators import DPTCoordinator
        
        coordinator = DPTCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
