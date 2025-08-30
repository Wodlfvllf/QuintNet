"""
Process group management for tensor parallelism.

This module provides utilities for managing distributed process groups
and organizing ranks into tensor parallel groups.
"""

import torch.distributed as dist


class ProcessGroupManager:
    """Manages process groups for tensor parallelism"""
    
    def __init__(self, tp_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert self.world_size % tp_size == 0, "world_size must be divisible by tp_size"

        # Identify which TP group this rank belongs to
        group_id = self.rank // tp_size
        ranks = list(range(group_id * tp_size, (group_id + 1) * tp_size))

        self.tp_group = dist.new_group(ranks=ranks)

    def get_group(self):
        """Get the tensor parallel process group"""
        return self.tp_group

    def get_tp_rank(self):
        """Get rank within the tensor parallel group"""
        return dist.get_rank(self.tp_group)

    def get_tp_world_size(self):
        """Get world size of the tensor parallel group"""
        return dist.get_world_size(self.tp_group)