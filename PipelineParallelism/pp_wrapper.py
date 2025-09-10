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


class PipelineStage:
    def __init__(self, module: nn.Module, rank: int, world_size: int, stage_idx: int, num_stages: int):
        self.module = module.to(next(module.parameters()).device if any(p.requires_grad for p in module.parameters()) else torch.device("cpu"))
        self.rank = rank
        self.world_size = world_size
        self.stage_idx = stage_idx
        self.num_stages = num_stages

    def forward(self, input_tensor=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # First stage: compute and send to next
        if self.stage_idx == 0:
            assert input_tensor is not None, "First stage must be given input_tensor"
            out = self.module(input_tensor)
            if self.num_stages > 1:
                # send out to next rank (stage)
                Send.apply(out, self.rank + 1)
            return out

        # Last stage: receive from previous, compute output and return
        if self.stage_idx == self.num_stages - 1:
            inp = Recv.apply(self.rank - 1, device)
            out = self.module(inp)
            return out

        # Middle stages: receive, compute, send
        inp = Recv.apply(self.rank - 1, device)
        out = self.module(inp)
        Send.apply(out, self.rank + 1)
        return out


class PipelineParallelWrapper:
    def __init__(self, model: nn.ModuleList, num_stages: int):
        """
        model: torch.nn.ModuleList of layers/blocks (must be ModuleList for easy slicing)
        num_stages: number of pipeline stages (assume world_size == num_stages)
        """
        assert dist.is_initialized(), "torch.distributed must be initialized (init_process_group)"
        self.model = model
        self.num_stages = num_stages
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert self.world_size == num_stages, "This simple wrapper assumes world_size == num_stages"
        self.stage_idx = self.rank  # direct mapping

        # divide the module list into `num_stages` contiguous chunks and keep only local stage
        stage_modules = self._divide_model_into_stages()
        self.local_module = stage_modules[self.stage_idx]
        self.pipeline_stage = PipelineStage(
            module=self.local_module,
            rank=self.rank,
            world_size=self.world_size,
            stage_idx=self.stage_idx,
            num_stages=self.num_stages
        )

    def _divide_model_into_stages(self):
        L = len(self.model)
        stages = []
        for i in range(self.num_stages):
            start_idx = (i * L) // self.num_stages
            end_idx = ((i + 1) * L) // self.num_stages
            if start_idx >= end_idx:
                # make sure each stage has at least one op (might need rework for very small L)
                end_idx = min(start_idx + 1, L)
            stage = nn.Sequential(*[self.model[j] for j in range(start_idx, end_idx)])
            stages.append(stage)
        return stages

    def forward(self, input_tensor=None):
        return self.pipeline_stage.forward(input_tensor)
