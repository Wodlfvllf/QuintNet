import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn
from .Processgroup import ProcessGroupManager

class PipelineParallelWrapper(nn.Module):
    def __init__(self, model: nn.ModuleList, pp_group):
        """
        model: torch.nn.ModuleList of layers/blocks (must be ModuleList for easy slicing)
        num_stages: number of pipeline stages (assume world_size == num_stages)
        """
        super(PipelineParallelWrapper, self).__init__()
        assert dist.is_initialized(), "torch.distributed must be initialized (init_process_group)"
        self.model = model
        self.pp_group = pp_group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size(pp_group)
        self.num_stages = self.world_size  # assume world_size == num_stages
        self.stage_idx = self.rank  # direct mapping

        # divide the module list into `num_stages` contiguous chunks and keep only local stage
        self.local_module = self._divide_model_into_stages()

    # Helper that builds a Sequential from a list of modules (flattens containers)
    def _make_stage_from_module_list(self, modules_list):
        result = []
        container_types = (nn.ModuleList, nn.ModuleDict, nn.ParameterList)

        def _append_or_flatten(m):
            if isinstance(m, container_types):
                for ch in m.children():
                    _append_or_flatten(ch)
                return
            if isinstance(m, nn.Sequential):
                # keep sequential as a single block
                result.append(m)
                return
            # if it has a forward, append
            if hasattr(m, "forward") and callable(getattr(m, "forward")):
                result.append(m)
                return
            # otherwise try to expand children (custom container)
            chs = list(m.children())
            if chs:
                for c in chs:
                    _append_or_flatten(c)
                return
            raise ValueError(f"Module {m} has no forward and no children; cannot include in Sequential")

        for mod in modules_list:
            _append_or_flatten(mod)

        if len(result) == 0:
            # return identity if nothing to do on this stage
            return nn.Identity()
        return nn.Sequential(*result)


    def _divide_model_into_stages(self):
        # 1) get modules as an ordered python list
        if isinstance(self.model, nn.ModuleList):
            modules = list(self.model)               # ModuleList supports iteration
        else:
            modules = list(self.model.children())    # fallback for other nn.Module wrappers

        total = len(modules)
        if total == 0:
            raise ValueError("Model contains no child modules to split")

        # 2) compute even split but give remainder to the LAST stage (so head stays on last)
        per = total // self.num_stages
        rem = total % self.num_stages

        # base size per stage, then give remainder to final stage
        sizes = [per] * self.num_stages
        sizes[-1] += rem

        # 3) compute start/end for this rank
        start = sum(sizes[:self.rank])
        end = start + sizes[self.rank]

        # If this stage would be empty, return Identity() (safe no-op)
        if start >= end:
            return nn.Identity()

        selected = modules[start:end]
        return self._make_stage_from_module_list(selected)
    
    def forward(self, input_tensor=None):
        return self.local_module(input_tensor)
    
    def train(self, mode=True):
        self.local_module.train(mode)
        return self
