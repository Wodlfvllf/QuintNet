"""Core DataParallel components."""

from .ddp import DataParallel
from .config import DistributedConfig, BucketConfig, ReductionStrategy

__all__ = ["DataParallel", "DistributedConfig", "BucketConfig", "ReductionStrategy"]