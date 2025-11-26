"""
Configuration Classes for DataParallel

This module defines dataclasses and enums for configuring the `DataParallel`
implementation. These configurations control various aspects of distributed
data parallelism, including gradient bucketing, reduction strategies, and
distributed environment settings.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The configuration classes provide a structured way to pass settings to the
`DataParallel` components.

-   **`ReductionStrategy`**: An Enum defining how gradients are aggregated
    (e.g., sum or mean).
-   **`BucketConfig`**: Specifies parameters for gradient bucketing, such as
    the maximum size of a bucket and whether to use `gradient_as_bucket_view`
    (an optimization for memory efficiency).
-   **`DistributedConfig`**: Holds essential information about the distributed
    environment, including the current rank, world size, and the communication
    process group.

These configurations allow for fine-grained control and experimentation with
different DDP behaviors.

===============================================================================
"""

import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ReductionStrategy(Enum):
    """
    Defines the strategy to be used for reducing gradients across ranks.

    Attributes:
        SUM (str): Gradients are summed across all ranks.
        MEAN (str): Gradients are averaged across all ranks (summed then divided by world size).
    """
    SUM = "sum"
    MEAN = "mean"

@dataclass
class BucketConfig:
    """
    Configuration for gradient bucketing in `DataParallel`.

    Attributes:
        capacity_mb (int): The maximum size of a gradient bucket in megabytes.
            Gradients are grouped until this capacity is reached.
        gradient_as_bucket_view (bool): If True, gradients are stored as views
            into the flattened bucket tensor, which can save memory.
        find_unused_parameters (bool): If True, DDP will attempt to find
            unused parameters in the backward pass. This is useful for models
            with conditional execution paths.
    """
    capacity_mb: int = 25
    gradient_as_bucket_view: bool = False
    find_unused_parameters: bool = False

@dataclass
class DistributedConfig:
    """
    Configuration for the distributed training environment within `DataParallel`.

    Attributes:
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes participating in the
            distributed training.
        process_group (Optional[dist.ProcessGroup]): The `torch.distributed`
            process group to be used for communication. If None, the default
            process group will be used.
        broadcast_buffers (bool): If True, model buffers (e.g., BatchNorm
            statistics) are broadcasted from rank 0 to all other ranks at the
            beginning of training.
    """
    rank: int = 0
    world_size: int = 1
    process_group: Optional[dist.ProcessGroup] = None
    broadcast_buffers: bool = True
