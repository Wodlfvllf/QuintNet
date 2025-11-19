import torch
import torch.distributed as dist
from typing import Tuple, Union
import math
import os
from datetime import timedelta

# ============================================================================
# EXAMPLES OF HOW THE MESH WORKS
# ============================================================================
#
# Example 1: Basic 3D Mesh with mesh_dim=(2, 2, 2)
# ------------------------------------------------
# When you call: init_mesh(mesh_dim=(2, 2, 2), mesh_name=('dp', 'pp', 'tp'))
#
# The mesh tensor looks like this (3D array of global ranks):
#
#        PP dimension (axis 1) →
#        ┌─────┬─────┐
#   DP   │ 0 1 │ 2 3 │  ← TP dimension (axis 2)
#   dim  ├─────┼─────┤
#   ↓    │ 4 5 │ 6 7 │
#        └─────┴─────┘
#
# Written as a 3D tensor:
# tensor([[[0, 1],   # dp=0, pp=0, tp=0-1
#          [2, 3]],  # dp=0, pp=1, tp=0-1
#
#         [[4, 5],   # dp=1, pp=0, tp=0-1
#          [6, 7]]]) # dp=1, pp=1, tp=0-1
#
# Process Groups Created:
# -----------------------
# DP groups (share same pp, tp): [[0,4], [1,5], [2,6], [3,7]]
#   - Rank 0 and 4 share dp group (both at pp=0, tp=0)
#   - Rank 2 and 6 share dp group (both at pp=1, tp=0)
#
# PP groups (share same dp, tp): [[0,2], [1,3], [4,6], [5,7]]
#   - Rank 0 and 2 share pp group (both at dp=0, tp=0)
#   - Rank 4 and 6 share pp group (both at dp=1, tp=0)
#
# TP groups (share same dp, pp): [[0,1], [2,3], [4,5], [6,7]]
#   - Rank 0 and 1 share tp group (both at dp=0, pp=0)
#   - Rank 6 and 7 share tp group (both at dp=1, pp=1)
#
# ============================================================================
#
# Example 2: What Rank 6 Sees
# ----------------------------
# When rank 6 calls _init_process_groups(), it discovers:
#   - Its DP group: [2, 6] (coordinates: pp=1, tp=0)
#   - Its PP group: [4, 6] (coordinates: dp=1, tp=0)
#   - Its TP group: [6, 7] (coordinates: dp=1, pp=1)
#
# self.groups = {
#     'dp': <ProcessGroup for ranks [2, 6]>,
#     'pp': <ProcessGroup for ranks [4, 6]>,
#     'tp': <ProcessGroup for ranks [6, 7]>
# }
#
# ============================================================================
#
# Example 3: Larger Mesh with mesh_dim=(4, 2, 2)
# -----------------------------------------------
# 4 DP replicas, 2 PP stages, 2 TP shards = 16 total GPUs
#
# mesh tensor shape: (4, 2, 2)
# Ranks 0-15 arranged as:
#
#         PP=0      PP=1
#      ┌──┬──┐  ┌──┬──┐
# DP=0 │ 0│ 1│  │ 2│ 3│  ← TP dimension
# DP=1 │ 4│ 5│  │ 6│ 7│
# DP=2 │ 8│ 9│  │10│11│
# DP=3 │12│13│  │14│15│
#      └──┴──┘  └──┴──┘
#
# DP groups (16 groups of 4 ranks each):
#   [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
#
# PP groups (8 groups of 2 ranks each):
#   [0,2], [1,3], [4,6], [5,7], [8,10], [9,11], [12,14], [13,15]
#
# TP groups (8 groups of 2 ranks each):
#   [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]
#
# ============================================================================
#
# Example 4: Using the API in Training Code
# ------------------------------------------
# mesh = init_mesh(mesh_dim=(2, 2, 2), mesh_name=('dp', 'pp', 'tp'))
#
# # Get my coordinates
# my_rank = dist.get_rank()  # e.g., rank 6
# coords = mesh.get_coordinates_tensor_search(my_rank)
# # Returns: [1, 1, 0] meaning dp=1, pp=1, tp=0
#
# # Get process groups for communication
# dp_group = mesh.get_group('dp')  # For gradient averaging across data parallel
# pp_group = mesh.get_group('pp')  # For pipeline stage communication
# tp_group = mesh.get_group('tp')  # For tensor sharding within a layer
#
# # Example: Reduce gradients across DP group
# dist.all_reduce(grad_tensor, group=dp_group)
#
# # Example: Send activations to next pipeline stage
# dist.send(activation, dst=next_rank, group=pp_group)
#
# ============================================================================


class MeshGenerator:
    def __init__(self,
                 mesh: Union[torch.Tensor, "ArrayLike"],
                 device_type: str = 'cuda',
                 mesh_dim: Tuple[int, ...] = (2,2,2),
                 mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp'),
                 is_backend_initialised: bool = False
                ):
        """
        Initialize the mesh and create process groups for each dimension.
        
        Args:
            mesh: The master map tensor showing which global rank is at each coordinate
            device_type: Device type ('cuda' for GPUs)
            mesh_dim: Dimensions of the mesh (e.g., (2, 2, 2) for 8 GPUs)
            mesh_name: Names for each dimension (e.g., ('dp', 'pp', 'tp'))
            is_backend_initialised: Whether dist.init_process_group was already called
        
        Example:
            # For 8 GPUs with 2x2x2 mesh:
            mesh = torch.arange(8).view(2, 2, 2)
            # Creates tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
            generator = MeshGenerator(mesh, 'cuda', (2,2,2), ('dp','pp','tp'))
        """

        if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
            raise ValueError(f"`mesh` must be a CPU tensor, got {mesh}")

        self.mesh = (
            mesh.detach().to(dtype=torch.int)
            if isinstance(mesh, torch.Tensor)
            else torch.tensor(mesh, device="cpu", dtype=torch.int)
        )
        self.dp_size = mesh_dim[0]
        self.pp_size = mesh_dim[1]
        self.tp_size = mesh_dim[2]        
        self.device_type = device_type
        self.mesh_dim = mesh_dim
        self.mesh_name = mesh_name
        
        self.groups = {} # Will be populated by _init_process_groups

        if not is_backend_initialised:
            self._setup_group_and_device()

        # This needs to be called after setup
        self._init_process_groups()
        
    def _setup_group_and_device(self):
        """
        Initialize the distributed backend and set the device for this process.
        
        This is the global "handshake" where all processes connect.
        Each process also claims its physical GPU (e.g., GPU 0, 1, 2, etc.)
        """

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

        # Now it's safe to call these
        world_size = dist.get_world_size()
        
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size {world_size}, but found {self.mesh.numel()} ranks!"
            )

        # CORRECTED: Simplified device setting logic
        device_module = getattr(torch, self.device_type, None)
        if device_module:
            local_rank = int(os.environ["LOCAL_RANK"])
            device_module.set_device(local_rank)

    def _init_process_groups(self):
        """
        Create process groups for each dimension and populate self.groups.
        
        Example walkthrough for rank 6 with mesh_dim=(2,2,2):
        
        Iteration 1 (dim=0, 'dp'):
          - Extracts DP groups: [[0,4], [1,5], [2,6], [3,7]]
          - Rank 6 finds itself in [2,6]
          - Creates group and stores: self.groups['dp'] = <ProcessGroup[2,6]>
        
        Iteration 2 (dim=1, 'pp'):
          - Extracts PP groups: [[0,2], [1,3], [4,6], [5,7]]
          - Rank 6 finds itself in [4,6]
          - Stores: self.groups['pp'] = <ProcessGroup[4,6]>
        
        Iteration 3 (dim=2, 'tp'):
          - Extracts TP groups: [[0,1], [2,3], [4,5], [6,7]]
          - Rank 6 finds itself in [6,7]
          - Stores: self.groups['tp'] = <ProcessGroup[6,7]>
        """
        my_rank = dist.get_rank()

        # CORRECTED: Use range(self.mesh.ndim) to iterate
        for dim in range(self.mesh.ndim):
            dim_name = self.mesh_name[dim]

            # Tensor magic: extract all groups for this dimension
            # Example for dim=0 (DP) with mesh shape (2,2,2):
            #   - swapdims(-1, 0) moves DP axis to the end
            #   - reshape(-1, 2) creates groups: [[0,4], [1,5], [2,6], [3,7]]
            process_groups_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
            
            # Search for which group this rank belongs to
            for mesh_dim in process_groups_by_dim:
                subgroup_ranks = mesh_dim.tolist()

                # Added the crucial check to find which group this rank belongs to
                if my_rank in subgroup_ranks:
                    # Found our group! Create the process group.
                    # Example: rank 6 and rank 2 both call this with ranks=[2,6]
                    new_pg = dist.new_group(
                        ranks=subgroup_ranks,
                        backend="nccl"
                    )
                    self.groups[dim_name] = new_pg
                    
                    # Break only after finding our group for this dimension
                    break
    
    def get_group(self, dim_name: str) -> dist.ProcessGroup:
        """
        Get the process group for a specific dimension.
        
        Args:
            dim_name: Name of the dimension ('dp', 'pp', or 'tp')
        
        Returns:
            ProcessGroup object for communication within that dimension
        
        Example:
            # In training code for rank 6:
            mesh = init_mesh(mesh_dim=(2,2,2), mesh_name=('dp','pp','tp'))
            
            dp_group = mesh.get_group('dp')  # Returns ProcessGroup[2,6]
            # Use for: dist.all_reduce(gradients, group=dp_group)
            
            pp_group = mesh.get_group('pp')  # Returns ProcessGroup[4,6]
            # Use for: dist.send(activations, dst=4, group=pp_group)
            
            tp_group = mesh.get_group('tp')  # Returns ProcessGroup[6,7]
            # Use for: dist.all_gather(shard, group=tp_group)
        """
        return self.groups[dim_name]
    
    def get_group_by_global_rank(self, global_rank: str) -> str:
        """Get the name of the process group that a global rank belongs to."""
        ans: str = ""
        for dim in range(self.mesh.ndim):
            dim_name = self.mesh_name[dim]
            if global_rank in self.process_groups[dim_name]:
                ans = dim_name
                
        return ans
            
    def get_coordinates(self, rank):
        """Get the coordinates of a rank in the mesh."""
        dp_coord = rank//(self.pp_size*self.tp_size)
        tp_coord = ((rank) // (self.pp_size))%(self.tp_size)
        pp_coord = (rank) % (self.pp_size)
        return [dp_coord, tp_coord, pp_coord]
    
    def get_coordinates_tensor_search(self, rank):
        """
        Find coordinates by searching the mesh tensor.
        This is the RECOMMENDED method - it's always correct.
        
        Args:
            rank: Global rank to find (0 to world_size-1)
        
        Returns:
            List of coordinates [dp_coord, pp_coord, tp_coord]
        
        Example for rank=6 with mesh_dim=(2,2,2):
            mesh = tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
            
            Rank 6 is at position [1, 1, 0] in the tensor:
            - dp dimension (axis 0): index 1
            - pp dimension (axis 1): index 1  
            - tp dimension (axis 2): index 0
            
            Returns: [1, 1, 0]
            
            This means rank 6 is:
            - In DP replica 1 (out of 2)
            - In PP stage 1 (out of 2)
            - In TP shard 0 (out of 2)
        """
        return (self.mesh == rank).nonzero()[0].tolist()
        
        
def init_mesh(
    device_type: str = 'cuda',
    mesh_dim: Tuple[int, ...] = (2,2,2),
    mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
):
    if mesh_dim is None:
        raise RuntimeError(
                    "mesh_dim cannot be None",
                    "Provide a suitable value",
                )
        
    if mesh_name is None:
        raise RuntimeError(
                    "mesh_name cannot be None",
                    "Provide a suitable value",
                )
    
    if device_type != 'cuda':
        raise RuntimeError(
                    "Only device_type cuda ia accepted",
                )
    
    if len(mesh_dim) != len(mesh_name):
        raise RuntimeError(
            f"mesh_dim and mesh_name must have the same length, but got {len(mesh_dim)} and {len(mesh_name)}"
        )
        
    with torch.device('cpu'):
        mesh = torch.arange(math.prod(mesh_dim), dtype=torch.int).view(mesh_dim)
        
    device_mesh = MeshGenerator(
        mesh,
        device_type,
        mesh_dim,
        mesh_name
    )
    
    return device_mesh