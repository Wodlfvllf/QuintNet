"""
Coordinator for 2D Hybrid Parallelism (DP + PP).
"""

import torch.nn as nn
import torch.distributed as dist
from .main_coordinator import BaseCoordinator
from ..core.process_groups import ProcessGroupManager
from ..parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper
from ..parallelism.data_parallel.core.ddp import DataParallel

class DPPCoordinator(BaseCoordinator):
    """
    Coordinator for applying a 2D hybrid of Data and Pipeline Parallelism (DP+PP).

    This strategy is useful for models that are too large to fit on a single
    GPU. It first splits the model into stages across multiple GPUs (pipeline
    parallelism) and then replicates this pipeline across multiple nodes or
    sets of GPUs (data parallelism).
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the DPPCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            device: The CUDA device where the model stage will be placed.
            **kwargs: Catches any additional arguments.
        """
        super().__init__(model, pg_manager=pg_manager, config=config, device=device, **kwargs)

    def parallelize(self) -> nn.Module:
        """
        Applies DP+PP to the model in the correct order.

        The correct order of application is:
        1.  **Pipeline Parallelism (PP):** The model is first split into stages
            across the GPUs in the pipeline-parallel group.
        2.  **Data Parallelism (DP):** The entire pipeline is then replicated,
            and `DataParallel` is used to synchronize gradients across these replicas.

        Returns:
            nn.Module: The fully parallelized model.
        """
        global_rank = dist.get_rank()

        # --- 1. Apply Pipeline Parallelism ---
        pp_group = self.pg_manager.get_group('pp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]

        pp_model = PipelineParallelWrapper(
            self.model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=pp_size,
            device=self.device
        )

        # --- 2. Apply Data Parallelism ---
        dp_model = DataParallel(pp_model)
        
        return dp_model