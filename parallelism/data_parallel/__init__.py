"""
CustomDDP: A modular distributed data parallel implementation.
"""

from .core import DataParallel
from .components import GradientBucket, BucketManager, GradientReducer, ParameterBroadcaster
from .utils import create_local_ddp, create_distributed_ddp

# Main exports
__all__ = [
    "DataParallel",
    "GradientBucket", 
    "BucketManager",
    "GradientReducer",
    "ParameterBroadcaster",
    "create_local_ddp",
    "create_distributed_ddp",
]

__version__ = "0.1.0"