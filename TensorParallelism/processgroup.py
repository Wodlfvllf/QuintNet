import torch
import torch.nn as nn
import torch.distributed as dist

class ProcessGroupManager:
    def __init__(self, tp_size):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        assert world_size == tp_size, "world_size must be Equivalent to tp_size"
        
        # For pure TP: just make one group of all ranks
        tp_group = dist.new_group(ranks=list(range(world_size)))
        return tp_group
