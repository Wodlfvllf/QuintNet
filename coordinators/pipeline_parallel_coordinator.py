"""
Coordinator for Pipeline Parallelism.
"""

import torch.nn as nn
import torch.distributed as dist
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper

class PipelineParallelCoordinator(BaseCoordinator):
    """
    Coordinator for applying Pipeline Parallelism (PP).

    This coordinator wraps a model with the `PipelineParallelWrapper`, which
    splits the model's layers into stages and distributes them across multiple
    GPUs in the pipeline parallel dimension.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the PipelineParallelCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager, used to
                get the pipeline parallel communication group.
            config (dict): The configuration dictionary, used to get mesh info.
            device: The CUDA device where the model stage will be placed.
            **kwargs: Catches any additional arguments passed from the strategy.
        """
        super().__init__(model, pg_manager=pg_manager, config=config, device=device, **kwargs)

    def parallelize(self) -> nn.Module:
        """
        Applies Pipeline Parallelism to the model.

        This method identifies the current process's rank and group within the
        pipeline parallel dimension and then wraps the model in the
        `PipelineParallelWrapper`.

        Returns:
            nn.Module: A `PipelineParallelWrapper` instance that contains the
                local stage of the model for the current rank.
        """
        global_rank = dist.get_rank()
        
        # Get the pipeline parallelism (pp) process group.
        pp_group = self.pg_manager.get_group('pp')
        
        # Find this rank's specific coordinate and rank within the pp dimension.
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]

        # The wrapper handles splitting the model and moving the correct stage to the device.
        return PipelineParallelWrapper(
            self.model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=pp_size,
            device=self.device
        )