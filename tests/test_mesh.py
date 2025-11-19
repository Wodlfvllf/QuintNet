"""
Tests for Process Mesh Management.
"""

import pytest
import torch
import torch.distributed as dist
from QuintNet.core.mesh import MeshGenerator

@pytest.mark.world_size(4)
def test_mesh_creation_and_groups(distributed_env):
    """
    Tests the creation of a 2x2 mesh and verifies process groups.
    This test must be run with a world size of 4.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 4:
        pytest.skip("This test requires a world size of 4.")

    mesh_dim = (2, 2)
    mesh_name = ('dp', 'tp')
    
    mesh_tensor = torch.arange(world_size).view(mesh_dim)
    device_mesh = MeshGenerator(mesh_tensor, 'cuda', mesh_dim, mesh_name, is_backend_initialised=True)

    # Test DP groups
    dp_group = device_mesh.get_group('dp')
    dp_group_size = dist.get_world_size(group=dp_group)
    assert dp_group_size == 2
    
    # Ranks 0 and 2 are in one DP group, 1 and 3 are in another
    if rank in [0, 2]:
        local_rank = dist.get_rank(group=dp_group)
        assert local_rank in [0, 1]
    if rank in [1, 3]:
        local_rank = dist.get_rank(group=dp_group)
        assert local_rank in [0, 1]

    # Test TP groups
    tp_group = device_mesh.get_group('tp')
    tp_group_size = dist.get_world_size(group=tp_group)
    assert tp_group_size == 2

    # Ranks 0 and 1 are in one TP group, 2 and 3 are in another
    if rank in [0, 1]:
        local_rank = dist.get_rank(group=tp_group)
        assert local_rank in [0, 1]
    if rank in [2, 3]:
        local_rank = dist.get_rank(group=tp_group)
        assert local_rank in [0, 1]


@pytest.mark.world_size(8)
def test_3d_mesh_coordinates(distributed_env):
    """
    Tests the coordinate calculation for a 2x2x2 3D mesh.
    This test must be run with a world size of 8.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 8:
        pytest.skip("This test requires a world size of 8.")

    mesh_dim = (2, 2, 2)
    mesh_name = ('dp', 'pp', 'tp')

    mesh_tensor = torch.arange(world_size).view(mesh_dim)
    device_mesh = MeshGenerator(mesh_tensor, 'cuda', mesh_dim, mesh_name, is_backend_initialised=True)

    # Expected coordinates for each rank
    expected_coords = {
        0: [0, 0, 0], 1: [0, 0, 1],
        2: [0, 1, 0], 3: [0, 1, 1],
        4: [1, 0, 0], 5: [1, 0, 1],
        6: [1, 1, 0], 7: [1, 1, 1],
    }

    coords = device_mesh.get_coordinates_tensor_search(rank)
    assert coords == expected_coords[rank]

    # Test a specific group
    pp_group = device_mesh.get_group('pp')
    if rank in [0, 1, 4, 5]:
        # These ranks belong to the first two PP groups
        local_rank = dist.get_rank(group=pp_group)
        assert local_rank in [0, 1]