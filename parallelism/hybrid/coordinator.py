import torch
import torch.nn as nn
import torch.distributed as dist
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel import apply_tensor_parallel
from QuintNet.parallelism.pipeline_parallel import PipelineParallelWrapper
from QuintNet.parallelism.data_parallel import CustomDDP
import os

class HybridCoordinator:
    """
    Coordinates the application of 3D parallelism (DP, PP, TP) to a model.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict):
        """
        Initializes the HybridCoordinator.

        Args:
            model (nn.Module): The base model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager for distributed communication.
            config (dict): A configuration dictionary.
        """
        self.model = model
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
        tp_rank = coords[1]
        pp_rank = coords[2]
        
        # Move model to device
        self.model.to(device)

        # 1. Apply Tensor Parallelism
        tp_model = apply_tensor_parallel(
            self.model,
            self.config['mesh_dim'][1],
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
            self.config['mesh_dim'][2],
            device
        ).to(device)

        # 3. Apply Data Parallelism
        dp_model = CustomDDP(pp_model)

        return dp_model