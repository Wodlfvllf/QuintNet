"""
Coordinator for Tensor Parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.rewrite import apply_tensor_parallel

class TensorParallelCoordinator(BaseCoordinator):
    """
    Coordinator for applying Tensor Parallelism (TP).

    This coordinator uses the `apply_tensor_parallel` function to traverse the
    model's module tree and replace relevant layers (e.g., `nn.Linear`) with
    their tensor-parallel equivalents (`ColumnParallelLinear`, `RowParallelLinear`).
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the TensorParallelCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager, used to
                get the tensor parallel communication group.
            config (dict): The configuration dictionary, used to get mesh info.
            device: The CUDA device where the model replica will be placed.
            **kwargs: Catches any additional arguments passed from the strategy.
        """
        super().__init__(model, pg_manager=pg_manager, config=config, device=device, **kwargs)

    def parallelize(self) -> nn.Module:
        """
        Applies Tensor Parallelism to the model.

        This method identifies the current process's rank and group within the
        tensor parallel dimension and then calls the `apply_tensor_parallel`
        rewriting function to shard the model's weights.

        Returns:
            nn.Module: The tensor-parallel model with sharded layers.
        """
        global_rank = dist.get_rank()
        
        # Get the tensor parallelism (tp) process group.
        tp_group = self.pg_manager.get_group('tp')
        
        # Find this rank's specific coordinate and rank within the tp dimension.
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        tp_size = self.config['mesh_dim'][self.config['mesh_name'].index('tp')]

        self.model.to(self.device)
        
        # The rewrite function handles the actual replacement of layers.
        return apply_tensor_parallel(
            self.model,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            device=self.device
        )