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

    def _get_model_type(self, model):
        """Detect the type of model for intelligent splitting"""
        # Check if it's a Vision Transformer
        has_embedding = hasattr(model, 'embedding')
        has_blocks = hasattr(model, 'blocks')
        has_classification_head = hasattr(model, 'classification_head')
        
        if has_embedding and has_blocks and has_classification_head:
            return 'vit'
        
        # Check if it's a standard transformer
        if has_blocks:
            return 'transformer'
        
        # Default to generic
        return 'generic'
    
