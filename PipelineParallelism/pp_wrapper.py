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
            
        elif self.num_stages == 3:
            if self.stage_idx == 0:
                # First stage: embedding + first third of blocks
                modules = [model.embedding]
                first_third = num_blocks // 3
                modules.extend(model.blocks[:first_third])
                return nn.Sequential(*modules)
            elif self.stage_idx == 1:
                # Middle stage: middle blocks
                first_third = num_blocks // 3
                second_third = 2 * num_blocks // 3
                return nn.Sequential(*model.blocks[first_third:second_third])
            else:
                # Last stage: last blocks + classification head
                second_third = 2 * num_blocks // 3
                modules = list(model.blocks[second_third:])
                modules.append(model.classification_head)
                return nn.Sequential(*modules)

        else:
            # For 4 or more stages, distribute transformer blocks evenly
            # Stage 0: embedding
            # Stage 1 to N-2: transformer blocks
            # Stage N-1: classification head
            
            if self.stage_idx == 0:
                # First stage: only embedding
                return model.embedding
            elif self.stage_idx == self.num_stages - 1:
                # Last stage: only classification head
                return model.classification_head
            else:
                # Middle stages: distribute transformer blocks
                blocks_per_stage = num_blocks // (self.num_stages - 2)
                remainder = num_blocks % (self.num_stages - 2)
                
                # Calculate which blocks belong to this stage
                stage_in_blocks = self.stage_idx - 1  # Adjusted index for block stages
                start_idx = stage_in_blocks * blocks_per_stage + min(stage_in_blocks, remainder)
                end_idx = start_idx + blocks_per_stage + (1 if stage_in_blocks < remainder else 0)
                
                if start_idx >= num_blocks:
                    # If no blocks assigned, create identity
                    return nn.Identity()
                
                stage_blocks = model.blocks[start_idx:end_idx]
                if len(stage_blocks) == 1:
                    return stage_blocks[0]
                else:
                    return nn.Sequential(*stage_blocks)
                
    def _split_transformer_model(self, model):
        """Split a transformer model with blocks"""
        if not hasattr(model, 'blocks'):
            return self._generic_split(model)
        
        num_blocks = len(model.blocks)
        blocks_per_stage = num_blocks // self.num_stages
        remainder = num_blocks % self.num_stages
        
        start_idx = self.stage_idx * blocks_per_stage + min(self.stage_idx, remainder)
        stage_size = blocks_per_stage + (1 if self.stage_idx < remainder else 0)
        end_idx = start_idx + stage_size
        
        stage_modules = model.blocks[start_idx:end_idx]
        
        if len(stage_modules) == 1:
            return stage_modules[0]
        else:
            return nn.Sequential(*stage_modules)
        
    def _generic_split(self, model):
        """Generic split for unknown model types - preserves module boundaries"""
        # Get top-level children modules
        children = list(model.children())
        
        if len(children) == 0:
            # If no children, return the model itself to one stage
            if self.stage_idx == 0:
                return model
            else:
                return nn.Identity()
            
        # Distribute children across stages
        num_children = len(children)
        children_per_stage = num_children // self.num_stages
        remainder = num_children % self.num_stages
        
        start_idx = self.stage_idx * children_per_stage + min(self.stage_idx, remainder)
        stage_size = children_per_stage + (1 if self.stage_idx < remainder else 0)
        end_idx = start_idx + stage_size
        
        stage_children = children[start_idx:end_idx]
        
        if len(stage_children) == 0:
            return nn.Identity()
        elif len(stage_children) == 1:
            return stage_children[0]
        else:
            return nn.Sequential(*stage_children)
        
    def _split_at_points(self, model, split_points: List[str]):
        """Split model at specific named module boundaries"""
        # This method allows manual specification of split points
        # Useful for complex models where automatic splitting fails
        
        named_modules = dict(model.named_modules())
        
        # Verify split points exist
        for point in split_points:
            if point not in named_modules:
                raise ValueError(f"Split point '{point}' not found in model")
        
        # Create stages based on split points
        # Implementation would depend on specific requirements
        raise NotImplementedError("Manual split points not yet implemented")