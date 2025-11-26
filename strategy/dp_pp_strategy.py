import torch.nn as nn
from .base_strategy import BaseStrategy

class DataPipelineParallelStrategy(BaseStrategy):
    """
    Data and Pipeline Parallelism Strategy (DP+PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from ..coordinators import DPPCoordinator
        
        coordinator = DPPCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
