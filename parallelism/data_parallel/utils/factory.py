"""Factory functions for creating DataParallel instances."""

import torch
import torch.distributed as dist
from typing import Optional

from ..core import DataParallel
from ..core.config import DistributedConfig, BucketConfig
from ..backends.local_backend import LocalBackend
from ..backends.torch_backend import TorchDistributedBackend

def create_local_ddp(model: torch.nn.Module, bucket_config: Optional[BucketConfig] = None) -> DataParallel:
    """Create a DataParallel instance for local (non-distributed) training."""
    return DataParallel(
        model=model,
        distributed_config=DistributedConfig(),
        bucket_config=bucket_config or BucketConfig(),
        backend=LocalBackend(),
    )

def create_distributed_ddp(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
    bucket_config: Optional[BucketConfig] = None,
) -> DataParallel:
    """Create a DataParallel instance for distributed training."""
    return DataParallel(
        model=model,
        distributed_config=DistributedConfig(
            rank=rank,
            world_size=world_size,
            process_group=process_group,
        ),
        bucket_config=bucket_config or BucketConfig(),
        backend=TorchDistributedBackend(),
    )