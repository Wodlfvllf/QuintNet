"""
Distributed Mesh Management

This module provides the `MeshGenerator` class, which is central to defining
and managing the distributed topology (device mesh) for hybrid parallelism
strategies in QuintNet. It handles the creation of communication process groups
for Data Parallelism (DP), Pipeline Parallelism (PP), and Tensor Parallelism (TP).

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

A "device mesh" is a logical N-dimensional grid that maps global ranks to
specific roles within different parallelism dimensions. For example, in a
3D mesh (DP, PP, TP), each global rank has a unique coordinate (dp_coord, pp_coord, tp_coord).

The `MeshGenerator` performs the following key functions:
1.  **Initializes Distributed Backend**: Ensures `torch.distributed` is set up.
2.  **Creates Process Groups**: Based on the defined `mesh_dim` and `mesh_name`,
    it creates distinct `torch.distributed.ProcessGroup` objects for each
    parallelism dimension (DP, PP, TP). A rank belongs to one DP group, one PP group,
    and one TP group.
3.  **Coordinate Lookup**: Provides methods to find a global rank's coordinates
    within the mesh, which is crucial for determining its role in communication.

This module is a foundational piece for any distributed training setup in QuintNet,
as it establishes the communication infrastructure.

===============================================================================
EXAMPLES OF HOW THE MESH WORKS:
===============================================================================

Example 1: Basic 3D Mesh with mesh_dim=(2, 2, 2)
------------------------------------------------
When you call: `init_mesh(mesh_dim=(2, 2, 2), mesh_name=('dp', 'pp', 'tp'))`

The mesh tensor looks like this (3D array of global ranks):

       PP dimension (axis 1) →
       ┌─────┬─────┐
  DP   │ 0 1 │ 2 3 │  ← TP dimension (axis 2)
  dim  ├─────┼─────┤
  ↓    │ 4 5 │ 6 7 │
       └─────┴─────┘

Written as a 3D tensor:
tensor([[[0, 1],   # dp=0, pp=0, tp=0-1
         [2, 3]],  # dp=0, pp=1, tp=0-1

        [[4, 5],   # dp=1, pp=0, tp=0-1
         [6, 7]]]) # dp=1, pp=1, tp=0-1

Process Groups Created (for a specific rank, e.g., global rank 6):
-----------------------
-   **DP group for rank 6**: [2, 6] (ranks that share pp=1, tp=0)
-   **PP group for rank 6**: [4, 6] (ranks that share dp=1, tp=0)
-   **TP group for rank 6**: [6, 7] (ranks that share dp=1, pp=1)

===============================================================================

Example 2: Larger Mesh with mesh_dim=(4, 2, 2)
-----------------------------------------------
4 DP replicas, 2 PP stages, 2 TP shards = 16 total GPUs

mesh tensor shape: (4, 2, 2)
Ranks 0-15 arranged as:

        PP=0      PP=1
     ┌──┬──┐  ┌──┬──┐
DP=0 │ 0│ 1│  │ 2│ 3│  ← TP dimension
DP=1 │ 4│ 5│  │ 6│ 7│
DP=2 │ 8│ 9│  │10│11│
DP=3 │12│13│  │14│15│
     └──┴──┘  └──┴──┘

DP groups (4 groups of 4 ranks each):
  [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]

PP groups (8 groups of 2 ranks each):
  [0,2], [1,3], [4,6], [5,7], [8,10], [9,11], [12,14], [13,15]

TP groups (8 groups of 2 ranks each):
  [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]

===============================================================================

Example 3: Using the API in Training Code
------------------------------------------
(Note: `init_mesh` is typically called via `init_process_groups` in `core/process_groups.py`)

.. code-block:: python

    from QuintNet.core.process_groups import init_process_groups

    # Initialize the mesh for 8 GPUs (2 DP, 2 PP, 2 TP)
    pg_manager = init_process_groups(mesh_dim=(2, 2, 2), mesh_name=('dp', 'pp', 'tp'))

    my_global_rank = dist.get_rank()  # e.g., global rank 6
    
    # Get my coordinates within the mesh
    coords = pg_manager.get_coordinates_tensor_search(my_global_rank)
    # If my_global_rank is 6, coords will be [1, 1, 0] (dp=1, pp=1, tp=0)

    # Get process groups for communication
    dp_group = pg_manager.get_group('dp')  # For gradient averaging across data parallel
    pp_group = pg_manager.get_group('pp')  # For pipeline stage communication
    tp_group = pg_manager.get_group('tp')  # For tensor sharding within a layer

    # Example usage of a process group:
    # dist.all_reduce(grad_tensor, group=dp_group)
    # dist.send(activation, dst=next_rank, group=pp_group)
    # dist.all_gather(shard, group=tp_group)

===============================================================================
"""

import torch
import torch.distributed as dist
from typing import Tuple, Union
import math
import os
from datetime import timedelta

class MeshGenerator:
    """
    Generates and manages the distributed device mesh and associated process groups.

    This class is responsible for initializing the distributed backend,
    creating communication groups for each parallelism dimension (DP, PP, TP),
    and providing utilities to query rank coordinates within the mesh.
    """
    def __init__(self,
                 mesh: Union[torch.Tensor, "ArrayLike"],
                 device_type: str = 'cuda',
                 mesh_dim: Tuple[int, ...] = (2,2,2),
                 mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp'),
                 is_backend_initialised: bool = False
                ):
        """
        Initializes the MeshGenerator.

        Args:
            mesh (Union[torch.Tensor, ArrayLike]): A tensor (or array-like)
                representing the N-dimensional grid of global ranks. Each element
                is a global rank, and its position defines its coordinates.
                Must be a CPU tensor.
            device_type (str): The type of device being used (e.g., 'cuda').
            mesh_dim (Tuple[int, ...]): A tuple specifying the size of each
                dimension in the mesh (e.g., (dp_size, pp_size, tp_size)).
            mesh_name (Tuple[str, ...]): A tuple of strings, providing names
                for each dimension (e.g., ('dp', 'pp', 'tp')).
            is_backend_initialised (bool): If True, assumes `torch.distributed`
                has already been initialized externally. If False, this class
                will attempt to initialize it.
        
        Raises:
            ValueError: If `mesh` is not a CPU tensor.
            RuntimeError: If the mesh size exceeds the world size.
        """

        if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
            raise ValueError(f"`mesh` must be a CPU tensor, got {mesh.device.type}")

        self.mesh = (
            mesh.detach().to(dtype=torch.int)
            if isinstance(mesh, torch.Tensor)
            else torch.tensor(mesh, device="cpu", dtype=torch.int)
        )
        # Store individual dimension sizes for convenience
        self.dp_size = mesh_dim[0]
        self.pp_size = mesh_dim[1]
        self.tp_size = mesh_dim[2]        
        self.device_type = device_type
        self.mesh_dim = mesh_dim
        self.mesh_name = mesh_name
        
        self.groups = {} # Dictionary to store created process groups (e.g., {'dp': pg_dp, 'pp': pg_pp})

        # Initialize the distributed backend if not already done
        if not is_backend_initialised:
            self._setup_group_and_device()

        # Create the process groups for each dimension
        self._init_process_groups()
        
    def _setup_group_and_device(self):
        """
        Initializes the `torch.distributed` backend and sets the CUDA device
        for the current process.

        This method performs the global "handshake" for distributed communication
        and assigns a physical GPU to each process based on its `LOCAL_RANK`
        environment variable.
        """

        if not dist.is_initialized():
            # Initialize the default process group with NCCCL backend
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

        world_size = dist.get_world_size()
        
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size {world_size}, but found {self.mesh.numel()} ranks!"
            )

        # Set the CUDA device for the current process based on LOCAL_RANK
        device_module = getattr(torch, self.device_type, None)
        if device_module:
            local_rank = int(os.environ["LOCAL_RANK"])
            device_module.set_device(local_rank)

    def _init_process_groups(self):
        """
        Creates `torch.distributed.ProcessGroup` objects for each dimension
        defined in `self.mesh_name` (e.g., 'dp', 'pp', 'tp').

        Each process group consists of ranks that share the same coordinates
        along all *other* dimensions. For example, a DP group contains ranks
        that have the same PP and TP coordinates but different DP coordinates.
        """
        my_rank = dist.get_rank()

        # Iterate through each dimension (dp, pp, tp) to create its corresponding process groups
        for dim in range(self.mesh.ndim):
            dim_name = self.mesh_name[dim]

            # Use tensor manipulation to extract all possible subgroups for the current dimension.
            # Example: For dim=0 (DP) with mesh shape (2,2,2):
            #   - self.mesh.swapdims(-1, dim) moves the current dimension to the last axis.
            #   - .reshape(-1, self.mesh.size(dim)) flattens the other dimensions and
            #     groups ranks along the current dimension.
            #   Result: A 2D tensor where each row is a list of global ranks forming a subgroup.
            process_groups_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
            
            # Search for which subgroup the current rank belongs to
            for mesh_dim_group in process_groups_by_dim:
                subgroup_ranks = mesh_dim_group.tolist()

                if my_rank in subgroup_ranks:
                    # If the current rank is part of this subgroup, create a new process group.
                    # All ranks in `subgroup_ranks` will call `dist.new_group` with the
                    # same list of ranks, ensuring they join the same group.
                    new_pg = dist.new_group(
                        ranks=subgroup_ranks,
                        backend="nccl"
                    )
                    self.groups[dim_name] = new_pg
                    
                    # Once the group for this dimension is found and created, move to the next dimension.
                    break
    
    def get_group(self, dim_name: str) -> dist.ProcessGroup:
        """
        Retrieves the `torch.distributed.ProcessGroup` for a specified parallelism dimension.

        Args:
            dim_name (str): The name of the dimension (e.g., 'dp', 'pp', or 'tp').

        Returns:
            dist.ProcessGroup: The process group object for communication within that dimension.
        
        Raises:
            KeyError: If `dim_name` is not a recognized dimension.
        """
        return self.groups[dim_name]
    
    def get_coordinates_tensor_search(self, rank: int) -> list:
        """
        Finds the N-dimensional coordinates of a given global rank within the device mesh.

        This method is robust and recommended as it directly queries the `self.mesh` tensor.

        Args:
            rank (int): The global rank to find (0 to world_size-1).

        Returns:
            list: A list of integer coordinates, e.g., `[dp_coord, pp_coord, tp_coord]`.
        
        Example for rank=6 with mesh_dim=(2,2,2) and mesh_name=('dp', 'pp', 'tp'):
            `self.mesh` = tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
            
            Searching for rank 6:
            - `(self.mesh == rank).nonzero()` returns `tensor([[1, 1, 0]])`
            - `[0].tolist()` extracts `[1, 1, 0]`
            
            This means rank 6 is:
            - In DP replica 1 (out of 2)
            - In PP stage 1 (out of 2)
            - In TP shard 0 (out of 2)
        """
        # `(self.mesh == rank).nonzero()` returns the coordinates of the element
        # in the tensor that matches `rank`. `[0]` takes the first (and only) match.
        return (self.mesh == rank).nonzero()[0].tolist()