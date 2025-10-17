

import torch
import torch.distributed as dist
from typing import Tuple, Union
import math
import os
from datetime import timedelta

# It's good practice to have this at the top level
# from torch._C._distributed_c10d import _get_default_group

class MeshGenerator:
    def __init__(self,
                 mesh: Union[torch.Tensor, "ArrayLike"],
                 device_type: str = 'cuda',
                 mesh_dim: Tuple[int, ...] = (2,2,2),
                 mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp'),
                 is_backend_initialised: bool = False
                ):

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
        # CORRECTED: Call init_process_group first. Let it get rank/world_size from env.
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
        my_rank = dist.get_rank()

        # CORRECTED: Use range(self.mesh.ndim) to iterate
        for dim in range(self.mesh.ndim):
            dim_name = self.mesh_name[dim]
            
            process_groups_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
            # self.process_groups[dim_name] = list(process_groups_by_dim.flatten())
            
            for mesh_dim in process_groups_by_dim:
                subgroup_ranks = mesh_dim.tolist()

                # Added the crucial check to find which group this rank belongs to
                if my_rank in subgroup_ranks:
                    new_pg = dist.new_group(
                        ranks=subgroup_ranks,
                        backend="nccl"
                    )
                    self.groups[dim_name] = new_pg
                    
                    # Break only after finding our group for this dimension
                    break
    
    def get_group(self, dim_name: str) -> dist.ProcessGroup:
        """Simple accessor to get the process group for a dimension."""
        return self.groups[dim_name]
    
    def get_group_by_global_rank(self, global_rank: str) -> str:
        ans: str = ""
        for dim in range(self.mesh.ndim):
            dim_name = self.mesh_name[dim]
            if global_rank in self.process_groups[dim_name]:
                ans = dim_name
                
        return ans
            
    def get_coordinates(self, rank):
        dp_coord = rank//(self.pp_size*self.tp_size)
        tp_coord = ((rank) // (self.pp_size))%(self.tp_size)
        pp_coord = (rank) % (self.pp_size)
        return [dp_coord, tp_coord, pp_coord]
    
    def get_coordinates_tensor_search(self, rank):
        # self.mesh is the tensor torch.arange(N).view(dp, tp, pp)
        # This finds where the rank is located in the tensor
        # and returns its [row, col, depth] index.
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