import torch.nn as nn
from .base_strategy import BaseStrategy

class TensorPipelineParallelStrategy(BaseStrategy):
    """
    Tensor and Pipeline Parallelism Strategy (TP+PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from ..coordinators.tp_pp_coordinator import TPPCoordinator
        
        coordinator = TPPCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
