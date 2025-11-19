import torch.nn as nn
from QuintNet.strategy.base_strategy import BaseStrategy

class DataParallelStrategy(BaseStrategy):
    """
    Data Parallelism Strategy (DP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.data_parallel_coordinator import DataParallelCoordinator
        
        coordinator = DataParallelCoordinator(model, self.device)
        return coordinator.parallelize()
