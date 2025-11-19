import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.rewrite import apply_tensor_parallel

class TensorParallelCoordinator:
    """
    Coordinator for applying Tensor Parallelism.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device):
        """
        Initializes the TensorParallelCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            device: The device to move the model to.
        """
        self.model = model
        self.pg_manager = pg_manager
        self.config = config
        self.device = device

    def parallelize(self) -> nn.Module:
        """
        Applies Tensor Parallelism to the model.

        Returns:
            nn.Module: The tensor-parallel model.
        """
        global_rank = dist.get_rank()
        tp_group = self.pg_manager.get_group('tp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]

        self.model.to(self.device)
        return apply_tensor_parallel(
            self.model,
            tp_size=self.config['mesh_dim'][self.config['mesh_name'].index('tp')],
            tp_rank=tp_rank,
            tp_group=tp_group,
            device=self.device
        )
