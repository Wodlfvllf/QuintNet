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

    def _flatten_model_layers(self, module):
        """Recursively flatten all layers from a model, handling nested containers"""
        flattened_layers = []
        
        for child_module in module.children():
            if isinstance(child_module, (nn.ModuleList, nn.Sequential)):
                # If it's a container, recursively flatten it
                flattened_layers.extend(self._flatten_model_layers(child_module))
            else:
                # If it's a regular layer, add it directly
                flattened_layers.append(child_module)
        
        return flattened_layers


    def _extract_layers_from_model(self, model):
        """Extract and flatten all layers from model"""
        # Start with empty list and flatten everything
        flattened_layers = self._flatten_model_layers(model)
        
        if len(flattened_layers) == 0:
            raise ValueError("Could not extract any layers from model")
        
        # Convert to ModuleList for easy indexing
        layers = nn.ModuleList(flattened_layers)
        
        print(f"Extracted {len(layers)} layers from model:")
        for i, layer in enumerate(layers):
            print(f"  Layer {i}: {type(layer).__name__}")
        
        return layers


    def _divide_model_into_stages(self):
        """Divide flattened layers into stages"""
        total_layers = len(self.model_layers)
        
        if total_layers < self.num_stages:
            raise ValueError(f"Cannot split {total_layers} layers into {self.num_stages} stages.")
        
        # Simple division with remainder distribution
        base_size = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        # Calculate start index for this stage
        start_idx = self.stage_idx * base_size + min(self.stage_idx, remainder)
        
        # Calculate size for this stage (distribute remainder to first few stages)
        stage_size = base_size + (1 if self.stage_idx < remainder else 0)
        end_idx = start_idx + stage_size
        
        # Get layers for this stage
        stage_layers = self.model_layers[start_idx:end_idx]
        
        print(f"Rank {self.rank}: Stage {self.stage_idx} has layers [{start_idx}:{end_idx}] "
            f"({len(stage_layers)} layers)")
        
        # Print what layers are in this stage
        for i, layer in enumerate(stage_layers):
            global_idx = start_idx + i
            print(f"  Layer {global_idx}: {type(layer).__name__}")
        
        # Return appropriate structure
        if len(stage_layers) == 1:
            return stage_layers[0]
        else:
            return nn.Sequential(*stage_layers)


    def _print_splitting_summary(self):
        """Print summary of how layers are split across all stages"""
        total_layers = len(self.model_layers)
        base_size = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        print(f"\n=== Model Splitting Summary ===")
        print(f"Total layers: {total_layers}")
        print(f"Number of stages: {self.num_stages}")
        print(f"Base layers per stage: {base_size}")
        print(f"Stages with extra layer: {remainder}")
        
        # Show distribution for all stages
        for stage_id in range(self.num_stages):
            start_idx = stage_id * base_size + min(stage_id, remainder)
            stage_size = base_size + (1 if stage_id < remainder else 0)
            end_idx = start_idx + stage_size
            print(f"  Stage {stage_id}: layers [{start_idx}:{end_idx}] ({stage_size} layers)")
        
        print("=" * 40)
    
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
