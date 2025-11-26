"""
Process Group Management

This module provides the primary user-facing entry point for initializing the
distributed environment in QuintNet. It defines the `ProcessGroupManager` class,
which handles the creation of the device mesh and all necessary communication
groups for different parallelism strategies.

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

A user's training script initializes everything through the `init_process_groups`
factory function.

.. code-block:: python

    from QuintNet.core.process_groups import init_process_groups

    # In the main training script:
    
    # This single call creates the mesh and all DP, PP, and TP groups.
    pg_manager = init_process_groups(
        device_type='cuda',
        mesh_dim=(2, 2, 2),
        mesh_name=('dp', 'pp', 'tp')
    )
    
    # The pg_manager can then be passed to strategies and coordinators,
    # which can retrieve the groups they need.
    dp_group = pg_manager.get_group('dp')

===============================================================================
"""

import torch
import torch.distributed as dist
from typing import Tuple, Dict
import math
from .mesh import MeshGenerator

class ProcessGroupManager:
    """
    Manages the creation and retrieval of process groups for a given mesh.

    This class acts as a high-level wrapper around the `MeshGenerator`. It is
    the main entry point for the training pipeline to set up the distributed
    environment based on a specified mesh configuration.
    """
    def __init__(self,
                 device_type: str = 'cuda',
                 mesh_dim: Tuple[int, ...] = (2,2,2),
                 mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
                ):
        """
        Initializes the ProcessGroupManager.

        This involves creating a tensor that represents the device mesh and then
        instantiating the `MeshGenerator` to create the actual process groups.

        Args:
            device_type (str): The type of device being used (e.g., 'cuda').
            mesh_dim (Tuple[int, ...]): The dimensions of the device mesh,
                corresponding to the number of GPUs for each parallelism type.
            mesh_name (Tuple[str, ...]): The names of the mesh dimensions,
                e.g., ('dp', 'pp', 'tp').
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
            
        # Create the mesh tensor on CPU. It's a map of global ranks to their
        # coordinates in the N-dimensional mesh.
        with torch.device('cpu'):
            mesh = torch.arange(math.prod(mesh_dim), dtype=torch.int).view(mesh_dim)
            
        # The MeshGenerator does the heavy lifting of initializing the distributed
        # backend and creating the process groups based on the mesh tensor.
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
            dim_name (str): Name of the dimension (e.g., 'dp', 'pp', or 'tp').

        Returns:
            dist.ProcessGroup: The communication group for that dimension.
        """
        return self.device_mesh.get_group(dim_name)

    def get_all_groups(self) -> Dict[str, dist.ProcessGroup]:
        """
        Returns a dictionary of all created process groups.

        Returns:
            Dict[str, dist.ProcessGroup]: A mapping from dimension name to
                the corresponding process group.
        """
        return self.device_mesh.groups
    
    def get_coordinates_tensor_search(self, rank: int) -> list:
        """
        Finds the coordinates of a given global rank within the device mesh.

        Args:
            rank (int): The global rank to find.

        Returns:
            list: A list of coordinates, e.g., `[dp_coord, pp_coord, tp_coord]`.
        """
        return self.device_mesh.get_coordinates_tensor_search(rank)

    def print_mesh_info(self):
        """
        Prints the device mesh configuration and rank assignments.
        Only prints from global rank 0.
        """
        if dist.get_rank() == 0:
            print("\n" + "="*60)
            print("Device Mesh Configuration")
            print("="*60)
            print(f"Mesh Dimensions: {self.device_mesh.mesh_dim}")
            print(f"Mesh Names:      {self.device_mesh.mesh_name}")
            print("-" * 60)
            print("Global Rank | Coordinates")
            print("-" * 60)
            
            world_size = dist.get_world_size()
            for rank in range(world_size):
                coords = self.get_coordinates_tensor_search(rank)
                coord_str = ", ".join([f"{name}={val}" for name, val in zip(self.device_mesh.mesh_name, coords)])
                print(f"Rank {rank:3d}    | {coord_str}")
            print("="*60 + "\n")

def init_process_groups(
    device_type: str = 'cuda',
    mesh_dim: Tuple[int, ...] = (2,2,2),
    mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
) -> ProcessGroupManager:
    """
    A factory function to initialize the ProcessGroupManager.

    This is the recommended entry point for users of the framework.

    Args:
        device_type (str): The type of device being used (e.g., 'cuda').
        mesh_dim (Tuple[int, ...]): The dimensions of the device mesh.
        mesh_name (Tuple[str, ...]): The names of the mesh dimensions.

    Returns:
        ProcessGroupManager: An initialized instance of the manager.
    """
    return ProcessGroupManager(device_type, mesh_dim, mesh_name)