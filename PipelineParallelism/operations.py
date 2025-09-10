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