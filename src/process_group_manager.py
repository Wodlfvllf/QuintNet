

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union
import math

def _get_device(device_type: str = "cuda"):
    """
    Get the module corresponding to the device_type which is cuda or cuda-like device.
    For example, when the device_type is cuda, the module `torch.cuda` is returned.
    Return None when there is no corresponding module for device_type, otherwise
    return the corresponding module.
    """
    return getattr(torch, device_type, None)
    
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
        
        self.device_type = device_type
        self.mesh_dim = mesh_dim
        self.mesh_name = mesh_name
        
        if not is_backend_initialised:
            #setup world group and device
            self._setup_group_and_device()
            
            #initialise each process groups
            self._init_process_groups()
            
        #implement getitem function to allow slicing of mesh
        
    def _setup_group_and_device(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=60)  # Longer timeout for NCCL
            )
        
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                    f"Mesh should not be bigger than default world size {world_size}, but found {self.mesh.numel()} ranks!"
                )
            
        device = _get_device()
        if device and not device.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
            logger.info(
                "Setting default device for the current process based on LOCAL_RANK=%s",
                local_rank,
            )
            device_handle.set_device(local_rank)
        
        return _get_default_group()
    
    def _init_process_groups(self):
        self.groups = {}
        default_group = _get_default_group()
        if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
            ranks = list(range(dist.get_world_size()))
            dim_group_names.append(dim_group.group_name)
        else:
            for dim in self.mesh.ndim:
                process_groups_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                for mesh_dim in process_groups_by_dim:
                    sub_group_ranks = mesh_dim.tolist()
                    new_pg = dist.new_group(
                        ranks=subgroup_ranks,
                        backend="nccl"
                    )
                    # Store the created group object in our dictionary.
                    self.groups[dim_name] = new_pg

                    # IMPORTANT: Since we found our group for this dimension,
                    # we can stop searching and move to the next dimension.
                    break
                
        
        
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
    
    if len(set(mesh_dim)) != len(mesh_name):
        raise RuntimeError(
                    "Each mesh_dim_name must be unique.",
                    f"Found repeated mesh_name in mesh_name {mesh_name}",
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