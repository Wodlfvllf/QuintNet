"""
Coordinator for Full 3D Hybrid Parallelism (DP + PP + TP).
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
from .main_coordinator import BaseCoordinator
from ..core import ProcessGroupManager
from ..parallelism import TensorParallel
from ..parallelism import PipelineParallelWrapper
from ..parallelism import DataParallel

class Hybrid3DCoordinator(BaseCoordinator):
    """
    Coordinator for applying full 3D hybrid parallelism (DP+PP+TP).

    This is the most advanced strategy, combining all three forms of parallelism.
    It is designed for training extremely large models on large-scale, multi-node
    GPU clusters.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, **kwargs):
        """
        Initializes the Hybrid3DCoordinator.

        Args:
            model (nn.Module): The base model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            **kwargs: Catches any additional arguments.
        """
        super().__init__(model, pg_manager=pg_manager, config=config, **kwargs)

    def parallelize(self) -> nn.Module:
        """
        Applies 3D parallelism to the model in the correct order.

        The correct order of application is:
        1.  **Tensor Parallelism (TP):** The model's layers are first sharded
            across the GPUs in the tensor-parallel group.
        2.  **Pipeline Parallelism (PP):** The tensor-sharded model is then
            split into stages across the pipeline-parallel group.
        3.  **Data Parallelism (DP):** The entire tensor-and-pipeline-parallel
            model is then replicated, and `DataParallel` is used to synchronize
            gradients across these replicas.

        Returns:
            nn.Module: The fully parallelized model.
        """
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")

        # Get all the necessary info from the process group manager
        tp_group = self.pg_manager.get_group('tp')
        pp_group = self.pg_manager.get_group('pp')

        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        
        self.model.to(device)

        # --- 1. Apply Tensor Parallelism ---
        tp_model = TensorParallel(
            self.model,
            self.config['mesh_dim'][self.config['mesh_name'].index('tp')],
            tp_rank,
            tp_group,
            device,
            gather_output=True,
            sync_gradients=True,
            method_of_parallelism="column"
        )

        # --- 2. Apply Pipeline Parallelism ---
        pp_model = PipelineParallelWrapper(
            tp_model, 
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            self.config['mesh_dim'][self.config['mesh_name'].index('pp')],
            device
        ).to(device)

        # --- 3. Apply Data Parallelism ---
        dp_model = DataParallel(pp_model)

        return dp_model