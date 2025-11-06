"""
Process Group Manager for Pipeline Parallelism
Handles initialization and management of distributed process groups.
"""

import os
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """
    Manages distributed process groups for pipeline parallelism.
    Supports pipeline parallelism (PP) and optional tensor parallelism (TP).
    """
    def __init__(self, pp_size=None, tp_size=1):
        """
        Initialize the process group manager.
        
        Args:
            pp_size: Number of pipeline parallel stages (if None, uses world size)
            tp_size: Number of tensor parallel processes (default: 1, not implemented yet)
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating ProcessGroupManager")
        
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        
        # Pipeline parallelism configuration
        self.pp_size = pp_size if pp_size is not None else self.global_world_size
        self.tp_size = tp_size
        
        if self.global_world_size % (self.pp_size * self.tp_size) != 0:
            raise ValueError(
                f"World size ({self.global_world_size}) must be divisible by "
                f"pp_size * tp_size ({self.pp_size} * {self.tp_size})"
            )
        
        # Calculate ranks
        self.pp_rank = self.global_rank % self.pp_size
        
        # Create pipeline parallel process group
        self.pp_group = None
        self._create_pp_group()
        
        # Store information about pipeline stages
        self.pp_prev_rank = self.pp_rank - 1 if self.pp_rank > 0 else None
        self.pp_next_rank = self.pp_rank + 1 if self.pp_rank < self.pp_size - 1 else None
        self.pp_is_first_stage = (self.pp_rank == 0)
        self.pp_is_last_stage = (self.pp_rank == self.pp_size - 1)
        
        # For future DP support
        self.cp_dp_world_size = 1
        
        if self.pp_rank == 0:
            print(f"[ProcessGroupManager] Initialized:")
            print(f"  Global rank: {self.global_rank}/{self.global_world_size}")
            print(f"  PP rank: {self.pp_rank}/{self.pp_size}")
            print(f"  Is first stage: {self.pp_is_first_stage}")
            print(f"  Is last stage: {self.pp_is_last_stage}")
    
    def _create_pp_group(self):
        """Create pipeline parallel process group."""
        # For now, all ranks are in the same PP group
        ranks = list(range(self.pp_size))
        self.pp_group = dist.new_group(ranks=ranks, backend='nccl')
        
        if self.pp_rank == 0:
            print(f"[ProcessGroupManager] Created PP group with ranks: {ranks}")
    
    def get_pp_group(self):
        """Get the pipeline parallel process group."""
        return self.pp_group
    
    def get_pp_rank(self):
        """Get current pipeline parallel rank."""
        return self.pp_rank
    
    def get_pp_world_size(self):
        """Get pipeline parallel world size."""
        return self.pp_size
    
    def is_first_stage(self):
        """Check if this is the first pipeline stage."""
        return self.pp_is_first_stage
    
    def is_last_stage(self):
        """Check if this is the last pipeline stage."""
        return self.pp_is_last_stage
    
    def get_prev_rank(self):
        """Get the rank of the previous pipeline stage."""
        return self.pp_prev_rank
    
    def get_next_rank(self):
        """Get the rank of the next pipeline stage."""
        return self.pp_next_rank
    
    def destroy(self):
        """Cleanup process groups."""
        if dist.is_initialized():
            dist.destroy_process_group()
