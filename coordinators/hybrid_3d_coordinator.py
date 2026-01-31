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
from ..parallelism.data_parallel.core.config import DistributedConfig
from ..core import load_gpt2_distributed
from ..utils.GPT2 import GPT2Stage, GPT2Config

class Hybrid3DCoordinator(BaseCoordinator):
    """
    Coordinator for applying full 3D hybrid parallelism (DP+PP+TP).

    This is the most advanced strategy, combining all three forms of parallelism.
    It is designed for training extremely large models on large-scale, multi-node
    GPU clusters.
    """
    def __init__(self, 
                model: nn.Module, 
                pg_manager: ProcessGroupManager, 
                config: dict, 
                checkpoint_path: str = None, 
                is_staged: bool = False, 
                **kwargs):
        """
        Initializes the Hybrid3DCoordinator.

        Args:
            model (nn.Module): The base model to be parallelized.
            pg_manager (ProcessGroupManager): The process group manager.
            config (dict): The configuration dictionary.
            **kwargs: Catches any additional arguments.
        """
        super().__init__(model, 
                        pg_manager=pg_manager, 
                        config=config, 
                        checkpoint_path = checkpoint_path, 
                        is_staged = is_staged, 
                        **kwargs)

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

        if self.is_staged:
            return self._parallelize_staged()
        else:
            return self._parallelize_non_staged()
        
    def _parallelize_staged(self) -> nn.Module:
        """
        Applies 3D parallelism with distributed loading from checkpoint.
        
        Flow:
        1. Load sharded weights (each GPU loads only its portion)
        2. Build GPT2Stage with TP already applied
        3. Wrap with Pipeline Parallelism
        4. Wrap with Data Parallelism

        Returns:
            nn.Module: The fully parallelized model.
        """
        # ─────────────────────────────────────────────────────────────────
        # Get device and ranks
        # ─────────────────────────────────────────────────────────────────
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        
        # Get process groups
        tp_group = self.pg_manager.get_group('tp')
        pp_group = self.pg_manager.get_group('pp')
        
        # Get coordinates and sizes
        coords = self.pg_manager.get_coordinates_tensor_search(global_rank)
        tp_rank = coords[self.config['mesh_name'].index('tp')]
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        tp_size = self.config['mesh_dim'][self.config['mesh_name'].index('tp')]
        pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]
        
        # ─────────────────────────────────────────────────────────────────
        # Convert dict config to GPT2Config object
        # self.config has both training params (batch_size, mesh_dim) and
        # model params (model_config sub-dict). Extract model_config.
        # ─────────────────────────────────────────────────────────────────
        if 'model_config' in self.config:
            model_config = GPT2Config.from_dict(self.config['model_config'])
        else:
            # Fallback: try to create from top-level config or use defaults
            model_config = GPT2Config.from_dict(self.config)
        
        # ─────────────────────────────────────────────────────────────────
        # 1. Load sharded weights from checkpoint
        # Each GPU loads only its required portion (memory efficient!)
        # ─────────────────────────────────────────────────────────────────
        state_dict = load_gpt2_distributed(
            checkpoint_path=self.checkpoint_path,
            pg_manager=self.pg_manager,
            config=self.config,  # Dict for mesh params
            model_config=model_config,  # GPT2Config for model params
            device=device,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # 2. Build GPT2Stage from sharded state_dict
        # TP is applied within the stage (ColumnParallel/RowParallel layers)
        # ─────────────────────────────────────────────────────────────────
        stage = GPT2Stage.from_sharded_state_dict(
            state_dict=state_dict,
            config=model_config,  # GPT2Config object
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            pp_group=pp_group,  # For weight tying gradient sync
            device=device,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # 3. Wrap with Pipeline Parallelism
        # Pass pre-built stage instead of building from full model
        # ─────────────────────────────────────────────────────────────────
        pp_model = PipelineParallelWrapper(
            model=stage,
            device_mesh=self.pg_manager.device_mesh,
            rank=pp_rank,
            pp_group=pp_group,
            pp_size=pp_size,
            device=device,
            stage_module=stage,  # Use pre-built stage
        )
        
        # ─────────────────────────────────────────────────────────────────
        # 4. Wrap with Data Parallelism
        # CRITICAL: Use DP-specific process group to avoid deadlocks!
        # ─────────────────────────────────────────────────────────────────
        dp_group = self.pg_manager.get_group('dp')
        dp_config = DistributedConfig(
            rank=self.pg_manager.get_rank('dp'),
            world_size=self.pg_manager.get_world_size('dp'),
            process_group=dp_group,
            broadcast_buffers=True,
        )
        
        dp_model = DataParallel(pp_model, dp_config)
        return dp_model

    def _parallelize_non_staged(self) -> nn.Module:
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
        # CRITICAL: Pass the DP-specific process group to avoid deadlocks!
        # Without this, DataParallel uses the global process group (all 8 ranks),
        # but Stage 1 ranks are blocked waiting for pipeline communication.
        dp_group = self.pg_manager.get_group('dp')
        dp_rank = coords[self.config['mesh_name'].index('dp')]
        dp_size = self.config['mesh_dim'][self.config['mesh_name'].index('dp')]
        
        dp_config = DistributedConfig(
            rank=dp_rank,
            world_size=dp_size,
            process_group=dp_group
        )
        dp_model = DataParallel(pp_model, distributed_config=dp_config)

        return dp_model