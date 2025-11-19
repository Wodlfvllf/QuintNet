from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
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

class DataParallelStrategy(BaseStrategy):
    """
    Data Parallelism Strategy (DP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.data_parallel_coordinator import DataParallelCoordinator
        
        coordinator = DataParallelCoordinator(model, self.device)
        return coordinator.parallelize()

class TensorParallelStrategy(BaseStrategy):
    """
    Tensor Parallelism Strategy (TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.tensor_parallel_coordinator import TensorParallelCoordinator
        
        coordinator = TensorParallelCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()

class PipelineParallelStrategy(BaseStrategy):
    """
    Pipeline Parallelism Strategy (PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.pipeline_parallel_coordinator import PipelineParallelCoordinator
        
        coordinator = PipelineParallelCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()

class DataTensorParallelStrategy(BaseStrategy):
    """
    Data and Tensor Parallelism Strategy (DP+TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.dp_tp_coordinator import DPTCoordinator
        
        coordinator = DPTCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()

class DataPipelineParallelStrategy(BaseStrategy):
    """
    Data and Pipeline Parallelism Strategy (DP+PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.dp_pp_coordinator import DPPCoordinator
        
        coordinator = DPPCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()

class TensorPipelineParallelStrategy(BaseStrategy):
    """
    Tensor and Pipeline Parallelism Strategy (TP+PP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.tp_pp_coordinator import TPPCoordinator
        
        coordinator = TPPCoordinator(model, self.pg_manager, self.config, self.device)
        return coordinator.parallelize()

class Hybrid3DStrategy(BaseStrategy):
    """
    Full 3D Hybrid Parallelism Strategy (DP, PP, TP).
    """
    def apply(self, model: nn.Module) -> nn.Module:
        from QuintNet.coordinators.hybrid_3d_coordinator import Hybrid3DCoordinator

        # The coordinator will handle moving the model to the device
        coordinator = Hybrid3DCoordinator(model, self.pg_manager, self.config)
        return coordinator.parallelize()

def get_strategy(strategy_name: str, pg_manager: ProcessGroupManager, config: Dict[str, Any]) -> BaseStrategy:
    """
    Factory function to get a parallelism strategy instance.
    
    Args:
        strategy_name (str): The name of the strategy (e.g., "dp", "tp", "pp", "3d").
        pg_manager (ProcessGroupManager): The process group manager.
        config (Dict[str, Any]): The configuration dictionary.
        
    Returns:
        BaseStrategy: An instance of the requested parallelism strategy.
    """
    strategies = {
        "dp": DataParallelStrategy,
        "tp": TensorParallelStrategy,
        "pp": PipelineParallelStrategy,
        "dp_tp": DataTensorParallelStrategy,
        "dp_pp": DataPipelineParallelStrategy,
        "tp_pp": TensorPipelineParallelStrategy,
        "3d": Hybrid3DStrategy,
    }
    strategy_class = strategies.get(strategy_name.lower())
    
    if strategy_class:
        return strategy_class(pg_manager, config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies are: {list(strategies.keys())}")