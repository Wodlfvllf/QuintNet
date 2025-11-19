import torch
import torch.nn as nn
import torch.distributed as dist
import os
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel import apply_tensor_parallel
from QuintNet.parallelism.pipeline_parallel import PipelineParallelWrapper
from QuintNet.parallelism.data_parallel import CustomDDP

class Hybrid3DCoordinator(BaseCoordinator):
    """
    Coordinates the application of 3D parallelism (DP, PP, TP) to a model.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, **kwargs):
        """
        Initializes the Hybrid3DCoordinator.

        Args:
            model (nn.Module): The base model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager for distributed communication.
            config (dict): A configuration dictionary.
        """
        super().__init__(model, **kwargs)
        self.pg_manager = pg_manager
        self.config = config

    def parallelize(self) -> nn.Module:
        """
        Applies 3D parallelism to the model.

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
        
        # Move model to device
        self.model.to(device)

        # 1. Apply Tensor Parallelism
        tp_model = apply_tensor_parallel(
            self.model,
            self.config['mesh_dim'][self.config['mesh_name'].index('tp')],
            tp_rank,
            tp_group,
            device,
            gather_output=True,
            sync_gradients=True,
            method_of_parallelism="column"
        )

        # 2. Apply Pipeline Parallelism
        pp_model = PipelineParallelWrapper(
            tp_model, 
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            self.config['mesh_dim'][self.config['mesh_name'].index('pp')],
            device
        ).to(device)

        # 3. Apply Data Parallelism
        dp_model = CustomDDP(pp_model)

        return dp_model
