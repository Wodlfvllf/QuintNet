import torch.nn as nn
from .base_strategy import BaseStrategy

class PipelineParallelStrategy(BaseStrategy):
    """
    Pipeline Parallelism Strategy (PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from ..coordinators import PipelineParallelCoordinator
        
        coordinator = PipelineParallelCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
