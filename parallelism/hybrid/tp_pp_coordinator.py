import torch
import torch.nn as nn
import torch.distributed as dist
import os
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.rewrite import apply_tensor_parallel
from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper

class TPPCoordinator:
    """
    Coordinator for applying Tensor Parallelism and Pipeline Parallelism (TP+PP).
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device):
        """
        Initializes the TPPCoordinator.

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
        Applies TP+PP to the model.

        Returns:
            nn.Module: The parallelized model.
        """
        global_rank = dist.get_rank()
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)

        # 1. Apply Tensor Parallelism
        tp_group = self.pg_manager.get_group('tp')
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        
        self.model.to(self.device)
        tp_model = apply_tensor_parallel(
            self.model,
            tp_size=self.config['mesh_dim'][self.config['mesh_name'].index('tp')],
            tp_rank=tp_rank,
            tp_group=tp_group,
            device=self.device
        )

        # 2. Apply Pipeline Parallelism
        pp_group = self.pg_manager.get_group('pp')
        pp_rank = coords[self.config['mesh_name'].index('pp')]

        pp_model = PipelineParallelWrapper(
            tp_model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=self.config['mesh_dim'][self.config['mesh_name'].index('pp')],
            device=self.device
        )
        return pp_model
