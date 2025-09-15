"""
Process group management for tensor parallelism.

This module provides utilities for managing distributed process groups
and organizing ranks into tensor parallel groups.
"""

import torch.distributed as dist

import torch.distributed as dist

class ProcessGroupManager:
    """
    Manages all process groups for hybrid parallelism (Pipeline, Tensor, Data).
    """
    def __init__(self, pp_size: int, tp_size: int):
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        
        assert self.world_size % (pp_size * tp_size) == 0, \
            "World size must be divisible by pp_size * tp_size"
            
        dp_size = self.world_size // (pp_size * tp_size)
        
        self.pp_group = None
        self.tp_group = None
        self.dp_group = None

        # Create Data Parallel Groups
        # Ranks with the same (pp_rank, tp_rank) belong to the same DP group.
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = [k * (pp_size * tp_size) + i * tp_size + j for k in range(dp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.dp_group = group

        # Create Tensor Parallel Groups
        # Ranks with the same (dp_rank, pp_rank) belong to the same TP group.
        for i in range(dp_size):
            for j in range(pp_size):
                ranks = [i * (pp_size * tp_size) + j * tp_size + k for k in range(tp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.tp_group = group

        # Create Pipeline Parallel Groups
        # Ranks with the same (dp_rank, tp_rank) belong to the same PP group.
        for i in range(dp_size):
            for j in range(tp_size):
                ranks = [i * (pp_size * tp_size) + k * tp_size + j for k in range(pp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.pp_group = group

    def get_pp_group(self):
        """Returns the pipeline parallel process group."""
        return self.pp_group

    def get_tp_group(self):
        """Returns the tensor parallel process group."""
        return self.tp_group

    def get_dp_group(self):
        """Returns the data parallel process group."""
        return self.dp_group

    # You can also add helpers to get ranks within these groups
    def get_pp_rank(self):
        return dist.get_rank(self.pp_group)

    def get_tp_rank(self):
        return dist.get_rank(self.tp_group)
        
    def get_dp_rank(self):
        return dist.get_rank(self.dp_group)