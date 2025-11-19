import torch
import torch.nn as nn
import torch.distributed as dist
import os
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper
from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP

class DPPCoordinator(BaseCoordinator):
    """
    Coordinator for applying Data Parallelism and Pipeline Parallelism (DP+PP).
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the DPPCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            device: The device to move the model to.
        """
        super().__init__(model, **kwargs)
        self.pg_manager = pg_manager
        self.config = config
        self.device = device

    def parallelize(self) -> nn.Module:
        """
        Applies DP+PP to the model.

        Returns:
            nn.Module: The parallelized model.
        """
        global_rank = dist.get_rank()

        # 1. Apply Pipeline Parallelism
        pp_group = self.pg_manager.get_group('pp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]

        pp_model = PipelineParallelWrapper(
            self.model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=self.config['mesh_dim'][self.config['mesh_name'].index('pp')],
            device=self.device
        )

        # 2. Apply Data Parallelism
        dp_model = CustomDDP(pp_model)
        return dp_model
