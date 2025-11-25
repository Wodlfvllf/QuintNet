import torch.nn as nn
from .base_strategy import BaseStrategy

class Hybrid3DStrategy(BaseStrategy):
    """
    Full 3D Hybrid Parallelism Strategy (DP, PP, TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from ..coordinators.hybrid_3d_coordinator import Hybrid3DCoordinator

        coordinator = Hybrid3DCoordinator(model, self.pg_manager, self.config)
        return coordinator.parallelize()
