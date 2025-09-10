# corrected_pipeline.py  (minimal, self-contained stage + wrappers)
import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn

HEADER_SIZE = 8  # fixed header length to transmit shape metadata


class Send(Function):
    """
    Autograd-friendly send: forward sends a tensor to `dst`.
    backward receives the gradient tensor from `dst`.
    """
    @staticmethod
    def forward(ctx, tensor, dst):
        ctx.dst = int(dst)
        ctx.shape = tuple(tensor.size())

        # send a small header first: [ndims, dim0, dim1, ...] padded to HEADER_SIZE
        header = torch.zeros(HEADER_SIZE, dtype=torch.long, device=tensor.device)
        header[0] = len(ctx.shape)
        header[1:1 + len(ctx.shape)] = torch.tensor(ctx.shape, dtype=torch.long, device=tensor.device)
        dist.send(header, dst=ctx.dst)          # header
        dist.send(tensor.contiguous(), dst=ctx.dst)  # payload
        # return an empty tensor: the sender doesn't need the payload locally
        return torch.empty(0, device=tensor.device)

    @staticmethod
    def backward(ctx, grad_dummy):
        # grad_dummy is an empty tensor; we must receive the gradient from dst
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        header = torch.zeros(HEADER_SIZE, dtype=torch.long, device=device)
        # receive header
        dist.recv(header, src=ctx.dst)
        nd = int(header[0].item())
        shape = [int(header[i].item()) for i in range(1, 1 + nd)]
        grad = torch.empty(shape, device=device)
        dist.recv(grad, src=ctx.dst)
        # return gradient for the input, and None for the integer dst argument
        return grad, None


class Recv(Function):
    """
    Autograd-friendly recv: forward receives a tensor from `src`.
    backward sends the gradient back to `src`.
    """
    @staticmethod
    def forward(ctx, src, device):
        src = int(src)
        device = torch.device(device)
        # receive header first
        header = torch.zeros(HEADER_SIZE, dtype=torch.long, device=device)
        dist.recv(header, src=src)
        nd = int(header[0].item())
        shape = [int(header[i].item()) for i in range(1, 1 + nd)]
        tensor = torch.empty(shape, device=device)
        dist.recv(tensor, src=src)
        ctx.src = src
        ctx.shape = shape
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # send the gradient back to src
        header = torch.zeros(HEADER_SIZE, dtype=torch.long, device=grad_output.device)
        header[0] = len(ctx.shape)
        header[1:1 + len(ctx.shape)] = torch.tensor(ctx.shape, dtype=torch.long, device=grad_output.device)
        dist.send(header, dst=ctx.src)
        dist.send(grad_output.contiguous(), dst=ctx.src)
        # returns two Nones corresponding to forward inputs (src, device)
        return None, None

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

    def _make_stage_from_children(self, model: nn.Module, start_idx: int, end_idx: int, inclusive_end: bool = False) -> nn.Sequential:
        """
        Build an nn.Sequential stage from model.children() between start_idx and end_idx.
        - start_idx: Python-style start index (can be 0).
        - end_idx: Python-style end index (exclusive by default).
        - inclusive_end: if True then end_idx is treated as inclusive (end_idx included).
        Returns an nn.Sequential that is safe to call (i.e., every element has forward).
        This flattens nested ModuleList/ModuleDict/ParameterList automatically.
        """
        # Convert children generator to a list (important!)
        children = list(model.children())

        if inclusive_end:
            end_idx = end_idx + 1

        if not (0 <= start_idx < len(children)) or not (0 <= end_idx <= len(children)) or start_idx >= end_idx:
            raise IndexError(f"Invalid slice: start={start_idx}, end={end_idx}, #children={len(children)}")

        selected = children[start_idx:end_idx]

        # result modules to put into Sequential
        result_modules = []

        def _append_or_flatten(module: nn.Module):
            """
            If module is a non-callable container (ModuleList/ModuleDict/ParameterList),
            expand its children. Otherwise add the module itself if it has forward.
            For custom containers without forward but having children, recursively expand.
            """
            # Modules that are known containers without forward
            container_types = (nn.ModuleList, nn.ModuleDict, nn.ParameterList)

            if isinstance(module, container_types):
                # expand each child
                for ch in module.children():
                    _append_or_flatten(ch)
                return

            # If module is nn.Sequential (callable) we can keep it as one block.
            if isinstance(module, nn.Sequential):
                result_modules.append(module)
                return

            # If module has a forward method, keep it.
            if hasattr(module, "forward") and callable(getattr(module, "forward")):
                result_modules.append(module)
                return

            # If module has children (custom container without forward), expand them:
            child_list = list(module.children())
            if child_list:
                for ch in child_list:
                    _append_or_flatten(ch)
                return

            # Nothing to expand and no forward -> we cannot use it
            raise ValueError(f"Module {module} (type={type(module)}) has no forward and no submodules; cannot be called.")

        for m in selected:
            _append_or_flatten(m)

        # build Sequential from flattened list
        return nn.Sequential(*result_modules)


    def _divide_model_into_stages(self):
        model_as_a_list = nn.ModuleList(self.model.children())
        L = len(model_as_a_list) // self.num_stages
        stages = nn.ModuleList()
        start_idx = (self.rank * L) 
        end_idx = ((self.rank + 1) * L)
        if start_idx >= end_idx:
            # make sure each stage has at least one op (might need rework for very small L)
            end_idx = min(start_idx + 1, L)
            
        stage = self._make_stage_from_children(model_as_a_list, start_idx, end_idx, inclusive_end=False)
        return stage

    def forward(self, input_tensor=None):
        return self.local_module(input_tensor)
    
    def train(self, mode=True):
        self.local_module.train(mode)
        return self
