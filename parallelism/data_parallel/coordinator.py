import torch.nn as nn
import torch.distributed as dist
from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP

class DataParallelCoordinator:
    """
    Coordinator for applying Data Parallelism.
    """
    def __init__(self, model: nn.Module, device):
        """
        Initializes the DataParallelCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            device: The device to move the model to.
        """
        self.model = model
        self.device = device

    def parallelize(self) -> nn.Module:
        """
        Applies Data Parallelism to the model.

        Returns:
            nn.Module: The data-parallel model.
        """
        self.model.to(self.device)
        if dist.get_world_size() > 1:
            return CustomDDP(self.model)
        return self.model
