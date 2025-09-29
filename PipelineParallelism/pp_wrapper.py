import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager

class PipelineParallelWrapper(nn.Module):
    def __init__(self, model, pp_group, split_points: Optional[List[str]] = None):
        """
        model: The full model to be split across pipeline stages
        pp_group: process group for pipeline parallelism
        split_points: Optional list of module names to use as split boundaries
        """
        super(PipelineParallelWrapper, self).__init__()
        assert dist.is_initialized(), "torch.distributed must be initialized"
        
        self.pp_group = pp_group
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size
        self.stage_idx = self.rank
        
        # Store original model structure
        self.full_model = model
        
        # Intelligently split the model
        if split_points:
            self.local_module = self._split_at_points(model, split_points)
        else:
            self.local_module = self._intelligent_split(model)
        
        # Store info about first/last stages
        self.is_first_stage = (self.stage_idx == 0)
        self.is_last_stage = (self.stage_idx == self.num_stages - 1)
        
        print(f"Rank {self.rank}: Initialized stage {self.stage_idx}/{self.num_stages-1}")
        self._print_stage_info()

    
