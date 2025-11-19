"""
Coordinator for 2D Hybrid Parallelism (DP + TP).
"""

import torch.nn as nn
import torch.distributed as dist
from QuintNet.coordinators.main_coordinator import BaseCoordinator
from QuintNet.core.process_groups import ProcessGroupManager
from QuintNet.parallelism.tensor_parallel.rewrite import apply_tensor_parallel
from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP

class DPTCoordinator(BaseCoordinator):
    """
    Coordinator for applying a 2D hybrid of Data and Tensor Parallelism (DP+TP).

    This strategy is common for training large models that can fit on a single
    node but benefit from both data and tensor parallelism. It first applies
    tensor parallelism to shard the model's weights within a node, and then
    applies data parallelism to replicate the sharded model across different
    nodes or GPUs.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the DPTCoordinator.

        Args:
            model (nn.Module): The model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            device: The CUDA device where the model replica will be placed.
            **kwargs: Catches any additional arguments.
        """
        super().__init__(model, pg_manager=pg_manager, config=config, device=device, **kwargs)

    def parallelize(self) -> nn.Module:
        """
        Applies DP+TP to the model in the correct order.

        The correct order of application is crucial:
        1.  **Tensor Parallelism (TP):** The model's layers are first sharded
            across the GPUs in the tensor-parallel group.
        2.  **Data Parallelism (DP):** The now tensor-sharded model is then
            replicated across the data-parallel group using `CustomDDP`.

        Returns:
            nn.Module: The fully parallelized model.
        """
        global_rank = dist.get_rank()
        
        # --- 1. Apply Tensor Parallelism ---
        tp_group = self.pg_manager.get_group('tp')
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        tp_size = self.config['mesh_dim'][self.config['mesh_name'].index('tp')]
        
        self.model.to(self.device)
        tp_model = apply_tensor_parallel(
            self.model,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            device=self.device
        )

        # --- 2. Apply Data Parallelism ---
        # The CustomDDP wrapper will handle gradient synchronization across the DP group.
        dp_model = CustomDDP(tp_model)
        
        return dp_model