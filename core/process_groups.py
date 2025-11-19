import torch
import torch.distributed as dist
from typing import Tuple, Dict
import math
from QuintNet.core.mesh import MeshGenerator

class ProcessGroupManager:
    """
    Manages the creation and retrieval of process groups for a given mesh.
    This is the main entry point for the training pipeline.
    """
    def __init__(self,
                 device_type: str = 'cuda',
                 mesh_dim: Tuple[int, ...] = (2,2,2),
                 mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
                ):
        """
        Initializes the ProcessGroupManager.
        """
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
            
        self.device_mesh = MeshGenerator(
            mesh,
            device_type,
            mesh_dim,
            mesh_name
        )

    def get_group(self, dim_name: str) -> dist.ProcessGroup:
        """
        Get the process group for a specific dimension.
        
        Args:
            dim_name: Name of the dimension ('dp', 'pp', or 'tp')
        
        Returns:
            ProcessGroup object for communication within that dimension
        """
        return self.device_mesh.get_group(dim_name)

    def get_all_groups(self) -> Dict[str, dist.ProcessGroup]:
        """
        Returns all created process groups.
        """
        return self.device_mesh.groups
    
    def get_coordinates_tensor_search(self, rank):
        """
        Find coordinates by searching the mesh tensor.
        """
        return self.device_mesh.get_coordinates_tensor_search(rank)

def init_process_groups(
    device_type: str = 'cuda',
    mesh_dim: Tuple[int, ...] = (2,2,2),
    mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
):
    """
    Initializes the ProcessGroupManager and returns it.
    """
    return ProcessGroupManager(device_type, mesh_dim, mesh_name)
