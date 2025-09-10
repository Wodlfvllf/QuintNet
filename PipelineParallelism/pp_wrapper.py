import torch
import torch.distributed as dist
import torch.nn as nn
from .Processgroup import ProcessGroupManager

class PipelineParallelWrapper(nn.Module):
    def __init__(self, model, pp_group):
        """
        model: The full model to be split across pipeline stages
        pp_group: process group for pipeline parallelism
        """
        super(PipelineParallelWrapper, self).__init__()
        assert dist.is_initialized(), "torch.distributed must be initialized"
        
        self.pp_group = pp_group
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size
        self.stage_idx = self.rank
        
        # Convert model to ModuleList for proper splitting
        self.model_layers = self._extract_layers_from_model(model)
        
        # Divide model into stages
        self.local_module = self._divide_model_into_stages()
        
        # Store info about first/last stages
        self.is_first_stage = (self.stage_idx == 0)
        self.is_last_stage = (self.stage_idx == self.num_stages - 1)
        
        print(f"Rank {self.rank}: Initialized stage {self.stage_idx}/{self.num_stages-1}")

    def _extract_layers_from_model(self, model):
        """Extract layers from model into a ModuleList for splitting"""
        layers = nn.ModuleList()
        
        # Check if model has expected Vision Transformer structure
        if hasattr(model, 'patch_embedding'):
            # Add patch embedding as first layer
            layers.append(model.patch_embedding)
            
            # Add transformer blocks
            if hasattr(model, 'blocks'):
                for block in model.blocks:
                    layers.append(block)
            elif hasattr(model, 'layers'):
                for layer in model.layers:
                    layers.append(layer)
            else:
                # Try to find sequential blocks
                for name, module in model.named_children():
                    if 'block' in name.lower() or 'layer' in name.lower():
                        layers.append(module)
            
            # Add layer norm if exists
            if hasattr(model, 'ln') or hasattr(model, 'norm'):
                norm_layer = getattr(model, 'ln', None) or getattr(model, 'norm', None)
                if norm_layer:
                    layers.append(norm_layer)
            
            # Add classification head as last layer
            if hasattr(model, 'head') or hasattr(model, 'fc'):
                head = getattr(model, 'head', None) or getattr(model, 'fc', None)
                if head:
                    layers.append(head)
        else:
            # Fallback: treat model as sequential or extract all children
            if isinstance(model, nn.Sequential):
                layers = nn.ModuleList(list(model))
            elif isinstance(model, nn.ModuleList):
                layers = model
            else:
                # Extract all meaningful children
                for name, module in model.named_children():
                    layers.append(module)
        
        if len(layers) == 0:
            raise ValueError("Could not extract layers from model")
            
        print(f"Extracted {len(layers)} layers from model")
        return layers

    def _divide_model_into_stages(self):
        """Divide model layers into stages and return local stage"""
        total_layers = len(self.model_layers)
        
        # Calculate layers per stage
        layers_per_stage = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        # Distribute layers - give extra layers to later stages
        stage_sizes = [layers_per_stage] * self.num_stages
        for i in range(remainder):
            stage_sizes[-(i+1)] += 1
        
        # Calculate start and end indices for this stage
        start_idx = sum(stage_sizes[:self.stage_idx])
        end_idx = start_idx + stage_sizes[self.stage_idx]
        
        # Get layers for this stage
        stage_layers = self.model_layers[start_idx:end_idx]
        
        print(f"Rank {self.rank}: Stage {self.stage_idx} has layers [{start_idx}:{end_idx}] "
              f"({len(stage_layers)} layers)")
        
        # Return as Sequential
        if len(stage_layers) == 0:
            return nn.Identity()
        elif len(stage_layers) == 1:
            return stage_layers[0]
        else:
            return nn.Sequential(*stage_layers)
    
    def forward(self, x):
        """Forward pass through local stage"""
        return self.local_module(x)
    
    def parameters(self):
        """Return parameters of local module"""
        return self.local_module.parameters()
    
    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        self.local_module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        self.local_module.eval()
        return self
    
    def state_dict(self):
        """Return state dict of local module"""
        return self.local_module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict to local module"""
        return self.local_module.load_state_dict(state_dict)