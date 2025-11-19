import torch.nn as nn
from QuintNet.strategy.base_strategy import BaseStrategy

class TensorParallelStrategy(BaseStrategy):
    """
    Tensor Parallelism Strategy (TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.tensor_parallel_coordinator import TensorParallelCoordinator
        
        coordinator = TensorParallelCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
