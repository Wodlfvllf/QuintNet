"""Parameter broadcasting component."""

import torch.nn as nn

from QuintNet.parallelism.data_parallel.backends.base import DistributedBackend
from QuintNet.parallelism.data_parallel.core.config import DistributedConfig

class ParameterBroadcaster:
    """Handles parameter broadcasting."""
    
    def __init__(self, backend: DistributedBackend, distributed_config: DistributedConfig):
        self.backend = backend
        self.config = distributed_config
    
    def broadcast_parameters(self, model: nn.Module) -> None:
        """Broadcast model parameters from rank 0 to all ranks."""
        if not self.backend.is_initialized():
            return
        
        print(f"CustomDDP Rank {self.config.rank}: Broadcasting parameters from rank 0")
        
        for param in model.parameters():
            self.backend.broadcast_tensor(tensor=param.data, src=0, group=self.config.process_group)

        print(f"CustomDDP Rank {self.config.rank}: Parameter broadcast complete")