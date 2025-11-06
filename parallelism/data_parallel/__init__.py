"""
CustomDDP: A modular distributed data parallel implementation.
"""

from QuintNet.parallelism.data_parallel.core.ddp import CustomDDP
from QuintNet.parallelism.data_parallel.core.config import DistributedConfig, BucketConfig, ReductionStrategy
from QuintNet.parallelism.data_parallel.utils.factory import create_local_ddp, create_distributed_ddp

# Main exports
__all__ = [
    "CustomDDP",
    "DistributedConfig", 
    "BucketConfig",
    "ReductionStrategy",
    "create_local_ddp",
    "create_distributed_ddp",
]

__version__ = "0.1.0"