"""
Hybrid 3D Strategy (DP + PP + TP)

Supports two modes:
1. Non-staged (default): Full model is passed, weights are sliced at runtime
2. Staged (checkpoint loading): Weights are loaded from checkpoint, sharded per GPU
"""

import torch.nn as nn
from .base_strategy import BaseStrategy


class Hybrid3DStrategy(BaseStrategy):
    """
    Full 3D Hybrid Parallelism Strategy (DP, PP, TP).
    
    Args (via config):
        checkpoint_path: Path to safetensors checkpoint (for staged mode)
        is_staged: If True, use distributed loading from checkpoint
    """
    
    def __init__(self, pg_manager, config: dict, checkpoint_path: str = None, is_staged: bool = False):
        super().__init__(pg_manager, config)
        self.checkpoint_path = checkpoint_path
        self.is_staged = is_staged
    
    def apply(self, model: nn.Module = None) -> nn.Module:
        """
        Apply 3D parallelism to the model.
        
        Args:
            model: The model to parallelize. Can be None if is_staged=True
                   (model will be built from checkpoint).
        
        Returns:
            The parallelized model.
        """
        from ..coordinators import Hybrid3DCoordinator

        coordinator = Hybrid3DCoordinator(
            model=model,
            pg_manager=self.pg_manager,
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            is_staged=self.is_staged,
        )
        return coordinator.parallelize()
