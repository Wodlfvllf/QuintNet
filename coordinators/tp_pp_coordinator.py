"""
Coordinator for 2D Hybrid Parallelism (TP + PP).
"""

import torch.nn as nn
import torch.distributed as dist
from .main_coordinator import BaseCoordinator
from ..core.process_groups import ProcessGroupManager
from ..parallelism.tensor_parallel.rewrite import apply_tensor_parallel
from ..parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper

class TPPCoordinator(BaseCoordinator):
    """
    Coordinator for applying a 2D hybrid of Tensor and Pipeline Parallelism (TP+PP).

    This is a powerful strategy for very large models that do not fit on a
    single GPU. It combines tensor parallelism to split individual large layers
    across GPUs, and pipeline parallelism to split the sequence of layers into
    stages on different GPUs.
    """
    def __init__(self, model: nn.Module, pg_manager: ProcessGroupManager, config: dict, device, **kwargs):
        """
        Initializes the TPPCoordinator.

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
        Applies TP+PP to the model in the correct order.

        The correct order of application is:
        1.  **Tensor Parallelism (TP):** The model's layers are first sharded
            across the GPUs in the tensor-parallel group.
        2.  **Pipeline Parallelism (PP):** The tensor-sharded model is then
            split into stages across the pipeline-parallel group.

        Returns:
            nn.Module: The fully parallelized model.
        """
        global_rank = dist.get_rank()
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)

        # --- 1. Apply Tensor Parallelism ---
        tp_group = self.pg_manager.get_group('tp')
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

        # --- 2. Apply Pipeline Parallelism ---
        pp_group = self.pg_manager.get_group('pp')
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]

        pp_model = PipelineParallelWrapper(
            tp_model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            pp_size=pp_size,
            device=self.device
        )
        return pp_model