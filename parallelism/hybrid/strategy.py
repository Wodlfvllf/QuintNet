"""
Strategy Selection for Hybrid Parallelism.
"""

from enum import Enum
from typing import Tuple


class ParallelismStrategy(Enum):
    """Enum for parallelism strategy selection."""
    DATA_ONLY = "dp"
    TENSOR_ONLY = "tp"
    PIPELINE_ONLY = "pp"
    DATA_TENSOR = "dp_tp"
    DATA_PIPELINE = "dp_pp"
    TENSOR_PIPELINE = "tp_pp"
    FULL_3D = "dp_tp_pp"


class HybridParallel:
    """
    High-level interface for hybrid parallelism.
    
    TODO: Create clean API for users
    """
    
    @staticmethod
    def auto_select_strategy(
        world_size: int,
        model_size: int,
        batch_size: int
    ) -> Tuple[int, int, int]:
        """
        TODO: Implement automatic strategy selection
        
        Automatically determine best mesh dimensions based on:
        - Available GPUs
        - Model size
        - Batch size
        
        Returns:
            Tuple of (dp_size, tp_size, pp_size)
        """
        pass
