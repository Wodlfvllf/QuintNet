"""
Coordinator for Data Parallelism.
"""

import torch.nn as nn
import torch.distributed as dist
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP

class DataParallelCoordinator(BaseCoordinator):
    """
    Coordinator for applying Data Parallelism (DP).

    This coordinator wraps a model with the `CustomDDP` module, which handles
    the synchronization of gradients across multiple GPUs. It is the simplest
    form of parallelism.
    """
    def __init__(self, model: nn.Module, device, **kwargs):
        """
        Initializes the DataParallelCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            device: The CUDA device where the model replica will be placed.
            **kwargs: Catches any additional arguments passed from the strategy.
        """
        super().__init__(model, **kwargs)
        self.device = device

    def parallelize(self) -> nn.Module:
        """
        Applies Data Parallelism to the model.

        This method moves the model to the correct device and, if the world size
        is greater than 1, wraps it with `CustomDDP`.

        Returns:
            nn.Module: The data-parallel model. If world size is 1, it returns
                the original model on the correct device.
        """
        self.model.to(self.device)
        
        # DDP is only necessary when there are multiple processes.
        if dist.get_world_size() > 1:
            return CustomDDP(self.model)
        
        return self.model