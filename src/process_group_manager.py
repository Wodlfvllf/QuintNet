

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import math

class MeshGenerator:
    def __init__(self,
                mesh: Union[torch.Tensor, "ArrayLike"],
                device_type: str = 'cuda',
                mesh_dim: Tuple[int, ...] = (2,2,2),
                mesh_name: Tuple[str, ...] = ('dp', 'pp', 'tp')
                ):
        
        #setup world group and device
        
        #initialise each process groups
        
        #implement getitem function to allow slicing of mesh
        
        
        
        # self.tensor_parallel_size = tensor_parallel_size
        # self.pipeline_parallel_size = pipeline_parallel_size
        # self.world_size = world_size
        # assert self.world_size == self.tensor_parallel_size*self.pipeline_parallel_size, "Worlds size and dimensions of pipeline_parallel_size and pipeline_parallel_size doesn't match"
        
        # self.num_gpus = self.world_size
        # self.gpu_matrix = torch.arange(self.num_gpus).reshape(self.tensor_parallel_size, self.pipeline_parallel_size)
        
        # self.pipeline_parallel_group = [self.gpu_matrix[i][j] for i in range(self.tensor_parallel_size) for j in range(self.pipeline_parallel_size)]
        # self.tensor_parallel_size = [self.gpu_matrix[i][j] for i in range(self.pipeline_parallel_size) for j in range(self.tensor_parallel_size)]
        
        pass 
        
        
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