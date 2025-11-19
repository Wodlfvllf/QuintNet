import torch
import torch.nn as nn
import torch.distributed as dist
import os
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.rewrite import apply_tensor_parallel
from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP

class DPTCoordinator(BaseCoordinator):
    """
    Coordinator for applying Data Parallelism and Tensor Parallelism (DP+TP).
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the DPTCoordinator.

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
        Applies DP+TP to the model.

        Returns:
            nn.Module: The parallelized model.
        """
        global_rank = dist.get_rank()
        
        # 1. Apply Tensor Parallelism
        tp_group = self.pg_manager.get_group('tp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        
        self.model.to(self.device)
        tp_model = apply_tensor_parallel(
            self.model,
            tp_size=self.config['mesh_dim'][self.config['mesh_name'].index('tp')],
            tp_rank=tp_rank,
            tp_group=tp_group,
            device=self.device
        )

        # 2. Apply Data Parallelism
        dp_model = CustomDDP(tp_model)
        return dp_model
