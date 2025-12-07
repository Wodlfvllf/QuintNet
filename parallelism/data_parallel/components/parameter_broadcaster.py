"""
Parameter Broadcasting Component

This module defines the `ParameterBroadcaster` class, which is responsible
for ensuring that all model replicas in a Distributed Data Parallel (DDP)
setup start with identical parameters.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

In DDP, each process typically initializes its own model replica. To ensure
consistent training, it's crucial that all these replicas begin with the
exact same set of weights. The `ParameterBroadcaster` achieves this by
broadcasting the initial parameters from a designated source rank (usually
rank 0 within the group) to all other ranks.

This component uses the configured `DistributedBackend` to perform the
broadcast operation for each parameter in the model.

===============================================================================
"""

import torch.nn as nn
import torch.distributed as dist

from ..backends.base import DistributedBackend
from ..core.config import DistributedConfig

class ParameterBroadcaster:
    """
    Handles the broadcasting of model parameters from a source rank (typically rank 0)
    to all other ranks in a distributed data parallel group.

    This ensures that all model replicas start with identical weights.
    """
    
    def __init__(self, backend: DistributedBackend, distributed_config: DistributedConfig):
        """
        Initializes the ParameterBroadcaster.

        Args:
            backend (DistributedBackend): The distributed communication backend to use.
            distributed_config (DistributedConfig): Configuration for the
                distributed environment, including the process group and rank.
        """
        self.backend = backend
        self.config = distributed_config
    
    def broadcast_parameters(self, model: nn.Module) -> None:
        """
        Broadcasts the `data` of each parameter in the model from the first rank
        in the process group to all other ranks.

        Args:
            model (nn.Module): The model whose parameters need to be broadcasted.
        """
        if not self.backend.is_initialized():
            # If distributed backend is not initialized, no broadcasting is needed.
            return
        
        # Determine the source rank for broadcasting.
        # When using a process group, we need the GLOBAL rank of the first member
        # of that group (not global rank 0, which may not be in this group).
        if self.config.process_group is not None:
            # Get all global ranks in this process group and use the first one
            group_ranks = dist.get_process_group_ranks(self.config.process_group)
            src_global_rank = group_ranks[0]  # First rank in the group
        else:
            # Default to global rank 0 when using the global process group
            src_global_rank = 0
        
        # Print a message to indicate the start of broadcasting, only on the current rank.
        # print(f"DataParallel Rank {self.config.rank}: Broadcasting parameters from global rank {src_global_rank}", flush=True)
        
        # Iterate through all parameters of the model and broadcast their data.
        for param in model.parameters():
            self.backend.broadcast_tensor(tensor=param.data, src=src_global_rank, group=self.config.process_group)

        # Print a message to indicate completion, only on the current rank.
        # print(f"DataParallel Rank {self.config.rank}: Parameter broadcast complete")
