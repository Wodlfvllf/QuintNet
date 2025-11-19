import torch.nn as nn
from QuintNet.strategy.base_strategy import BaseStrategy

class PipelineParallelStrategy(BaseStrategy):
    """
    Pipeline Parallelism Strategy (PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.pipeline_parallel_coordinator import PipelineParallelCoordinator
        
        coordinator = PipelineParallelCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()
