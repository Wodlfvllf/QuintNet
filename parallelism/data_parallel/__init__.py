"""
Data Parallelism Implementation.

This module contains:
- Custom DDP implementation
- Gradient bucketing and reduction
- Parameter broadcasting
- Synchronization utilities

Migration Source: QuintNet/DataParallelsim/
"""

from QuintNet.parallelism.data_parallel.ddp import DataParallel, CustomDDP
from QuintNet.parallelism.data_parallel.components import (
    Bucket,
    BucketManager,
    GradientReducer,
    ParameterBroadcaster
)

__all__ = [
    'DataParallel',
    'CustomDDP',
    'Bucket',
    'BucketManager',
    'GradientReducer',
    'ParameterBroadcaster',
]
