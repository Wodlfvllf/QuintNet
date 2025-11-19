from typing import Dict, Any
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.strategy.base_strategy import BaseStrategy
from QuintNet.strategy.data_parallel_strategy import DataParallelStrategy
from QuintNet.strategy.tensor_parallel_strategy import TensorParallelStrategy
from QuintNet.strategy.pipeline_parallel_strategy import PipelineParallelStrategy
from QuintNet.strategy.dp_tp_strategy import DataTensorParallelStrategy
from QuintNet.strategy.dp_pp_strategy import DataPipelineParallelStrategy
from QuintNet.strategy.tp_pp_strategy import TensorPipelineParallelStrategy
from QuintNet.strategy.hybrid_3d_strategy import Hybrid3DStrategy

def get_strategy(strategy_name: str, pg_manager: ProcessGroupManager, config: Dict[str, Any]) -> BaseStrategy:
    """
    Factory function to get a parallelism strategy instance.
    
    Args:
        strategy_name (str): The name of the strategy.
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
