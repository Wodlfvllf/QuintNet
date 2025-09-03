"""Configuration classes for CustomDDP."""

import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ReductionStrategy(Enum):
    """Strategy for gradient reduction."""
    SUM = "sum"
    MEAN = "mean"

@dataclass
class BucketConfig:
    """Configuration for gradient bucketing."""
    capacity_mb: int = 25
    gradient_as_bucket_view: bool = False
    find_unused_parameters: bool = False

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    rank: int = 0
    world_size: int = 1
    process_group: Optional[dist.ProcessGroup] = None
    broadcast_buffers: bool = True