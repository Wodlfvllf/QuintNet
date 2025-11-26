"""
Strategy Module Entry Point

This module serves as the main entry point for accessing the various parallelism
strategies available in QuintNet. It exposes the `get_strategy` factory function,
which is the primary user-facing API for selecting and initializing a strategy.

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

A user's training script interacts with this module to get the desired
strategy object, which can then be used to parallelize the model.

.. code-block:: python

    from QuintNet.strategy import get_strategy

    # In the main training script:
    
    # 1. Initialize the process group manager and load config
    pg_manager = init_process_groups(config)
    
    # 2. Use the factory to get the strategy instance
    # The strategy name is typically read from the config file.
    strategy = get_strategy(
        config['strategy_name'],
        pg_manager,
        config
    )
    
    # 3. Apply the strategy to the model
    parallel_model = strategy.apply(model)

This abstracts away the internal details of which strategy class or coordinator
is being used.

===============================================================================
"""

from typing import Dict, Any
from .base_strategy import BaseStrategy
from .data_parallel_strategy import DataParallelStrategy
from .tensor_parallel_strategy import TensorParallelStrategy
from .pipeline_parallel_strategy import PipelineParallelStrategy
from .dp_tp_strategy import DataTensorParallelStrategy
from .dp_pp_strategy import DataPipelineParallelStrategy
from .tp_pp_strategy import TensorPipelineParallelStrategy
from .hybrid_3d_strategy import Hybrid3DStrategy
from ..core import ProcessGroupManager

def get_strategy(strategy_name: str, pg_manager: ProcessGroupManager, config: Dict[str, Any]) -> BaseStrategy:
    """
    Factory function to get a parallelism strategy instance.

    This function acts as a single entry point for creating any of the available
    parallelism strategies. It takes a string name and returns an initialized
    strategy object that is ready to be used.

    Args:
        strategy_name (str): The name of the strategy to be used.
            Valid options are: "dp", "tp", "pp", "dp_tp", "dp_pp", "tp_pp", "3d".
        pg_manager (ProcessGroupManager): The process group manager, which
            provides information about the distributed setup.
        config (Dict[str, Any]): The global configuration dictionary.

    Returns:
        BaseStrategy: An initialized instance of the requested parallelism strategy.
    
    Raises:
        ValueError: If the provided `strategy_name` is not a valid strategy.
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
        # Return an instance of the chosen strategy class
        return strategy_class(pg_manager, config)
    else:
        raise ValueError(f"Unknown strategy: '{strategy_name}'. Available strategies are: {list(strategies.keys())}")