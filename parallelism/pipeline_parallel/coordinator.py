import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper

class PipelineParallelCoordinator:
    """
    Coordinator for applying Pipeline Parallelism.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device):
        """
        Initializes the PipelineParallelCoordinator.

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
        Applies Pipeline Parallelism to the model.

        Returns:
            nn.Module: The pipeline-parallel model.
        """
        global_rank = dist.get_rank()
        pp_group = self.pg_manager.get_group('pp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]

        return PipelineParallelWrapper(
            self.model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=self.config['mesh_dim'][self.config['mesh_name'].index('pp')],
            device=self.device
        )
