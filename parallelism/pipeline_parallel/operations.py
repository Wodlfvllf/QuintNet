"""
Pipeline Parallel Communication Operations.

Migration Source: QuintNet/PipelineParallelism/operations.py
"""

import torch
from typing import Optional


def send_forward(
    tensor: torch.Tensor,
    dst: int,
    group: Optional[torch.distributed.ProcessGroup] = None
):
    """
    TODO: Migrate from QuintNet/PipelineParallelism/operations.py
    
    Send activation forward to next pipeline stage.
    """
    pass


def recv_forward(
    tensor_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    group: Optional[torch.distributed.ProcessGroup] = None
) -> torch.Tensor:
    """
    TODO: Migrate from QuintNet/PipelineParallelism/operations.py
    
    Receive activation from previous pipeline stage.
    """
    pass


def send_backward(
    tensor: torch.Tensor,
    dst: int,
    group: Optional[torch.distributed.ProcessGroup] = None
):
    """
    TODO: Migrate from QuintNet/PipelineParallelism/operations.py
    
    Send gradient backward to previous pipeline stage.
    """
    pass


def recv_backward(
    tensor_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    group: Optional[torch.distributed.ProcessGroup] = None
) -> torch.Tensor:
    """
    TODO: Migrate from QuintNet/PipelineParallelism/operations.py
    
    Receive gradient from next pipeline stage.
    """
    pass
