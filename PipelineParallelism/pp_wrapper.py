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
    
    def _intelligent_split(self, model):
        """Intelligently split model based on its architecture"""
        model_type = self._get_model_type(model)
        
        if model_type == 'vit':
            return self._split_vit_model(model)
        elif model_type == 'transformer':
            return self._split_transformer_model(model)
        else:
            return self._generic_split(model)
    
    def _split_vit_model(self, model):
        """Split Vision Transformer model intelligently"""
        # Vision Transformer has: embedding, transformer blocks, classification head
        
        # Count total components
        num_blocks = len(model.blocks) if hasattr(model, 'blocks') else 0
        total_components = 1 + num_blocks + 1  # embedding + blocks + head
        
        if self.num_stages == 1:
            return model
        
        if self.num_stages == 2:
            if self.stage_idx == 0:
                # First stage: embedding + half of transformer blocks
                modules = [model.embedding]
                half_blocks = num_blocks // 2
                modules.extend(model.blocks[:half_blocks])
                return nn.Sequential(*modules)
            else:
                # Second stage: rest of blocks + classification head
                half_blocks = num_blocks // 2
                modules = list(model.blocks[half_blocks:])
                modules.append(model.classification_head)
                return nn.Sequential(*modules)
