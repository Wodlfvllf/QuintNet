import torch
import torch.nn as nn
import torch.distributed as dist

class ProcessGroupManager:
    def __init__(self, tp_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert self.world_size % tp_size == 0, "world_size must be divisible by tp_size"

        # Identify which TP group this rank belongs to
        group_id = self.rank // tp_size
        ranks = list(range(group_id * tp_size, (group_id + 1) * tp_size))

        self.tp_group = dist.new_group(ranks=ranks)

    def get_group(self):
        return self.tp_group

    def get_tp_rank(self):
        return dist.get_rank(self.tp_group)

    def get_tp_world_size(self):
        return dist.get_world_size(self.tp_group)