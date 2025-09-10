"""
Process group management for tensor parallelism.

This module provides utilities for managing distributed process groups
and organizing ranks into tensor parallel groups.
"""

import torch.distributed as dist


class ProcessGroupManager:
    """Manages process groups for tensor parallelism"""
    
    def __init__(self, pp_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert self.world_size % pp_size == 0, "world_size must be divisible by pp_size"

        # Identify which PP group this rank belongs to
        group_id = self.rank // pp_size
        ranks = list(range(group_id * pp_size, (group_id + 1) * pp_size))

        self.pp_group = dist.new_group(ranks=ranks)

    def get_group(self):
        """Get the pipeline parallel process group"""
        return self.pp_group

    def get_pp_rank(self):
        """Get rank within the pipeline parallel group"""
        return dist.get_rank(self.pp_group)

    def get_pp_world_size(self):
        """Get world size of the pipeline parallel group"""
        return dist.get_world_size(self.pp_group)