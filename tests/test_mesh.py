"""
Tests for Process Mesh Management.

This module contains unit tests for the `MeshGenerator` class, ensuring
that the distributed device mesh is correctly created and that process
groups are properly formed according to the specified dimensions.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

The `MeshGenerator` is a foundational component for setting up hybrid
parallelism. These tests verify its core functionalities:
-   **Mesh Creation**: Checks if the `MeshGenerator` can be initialized
    with various mesh dimensions.
-   **Process Group Formation**: Verifies that `torch.distributed.ProcessGroup`
    objects are correctly created for each parallelism dimension (DP, PP, TP)
    and that each rank belongs to the expected groups.
-   **Coordinate Lookup**: Ensures that the `get_coordinates_tensor_search`
    method accurately returns the N-dimensional coordinates of a global rank
    within the mesh.

These tests are marked with `@pytest.mark.world_size(N)` to indicate the
required number of processes for execution, and they rely on the `distributed_env`
fixture from `conftest.py` to manage the distributed environment setup and teardown.

===============================================================================
"""

import pytest
import torch
import torch.distributed as dist
from QuintNet.core.mesh import MeshGenerator

@pytest.mark.world_size(4)
def test_mesh_creation_and_groups(distributed_env):
    """
    Tests the creation of a 2x2 mesh and verifies the formation of DP and TP process groups.

    This test requires a world size of 4.
    - Mesh: (dp=2, tp=2)
    - Global ranks:
      [[0, 1],
       [2, 3]]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != 4:
        pytest.skip("This test requires a world size of 4.")

    mesh_dim = (2, 2)
    mesh_name = ('dp', 'tp')
    
    # Create a mesh tensor representing the global ranks in the 2x2 grid
    mesh_tensor = torch.arange(world_size).view(mesh_dim)
    # Initialize the MeshGenerator, assuming distributed backend is already set up by fixture
    device_mesh = MeshGenerator(mesh_tensor, 'cuda', mesh_dim, mesh_name, is_backend_initialised=True)

    # --- Test DP groups ---
    dp_group = device_mesh.get_group('dp')
    dp_group_size = dist.get_world_size(group=dp_group)
    assert dp_group_size == 2 # Each DP group should have 2 ranks
    
    # Verify local ranks within DP groups
    # DP groups: [0, 2] and [1, 3]
    if rank in [0, 2]: # Ranks in the first DP group
        local_rank = dist.get_rank(group=dp_group)
        assert local_rank in [0, 1]
    if rank in [1, 3]: # Ranks in the second DP group
        local_rank = dist.get_rank(group=dp_group)
        assert local_rank in [0, 1]

    # --- Test TP groups ---
    tp_group = device_mesh.get_group('tp')
    tp_group_size = dist.get_world_size(group=tp_group)
    assert tp_group_size == 2 # Each TP group should have 2 ranks

    # Verify local ranks within TP groups
    # TP groups: [0, 1] and [2, 3]
    if rank in [0, 1]: # Ranks in the first TP group
        local_rank = dist.get_rank(group=tp_group)
        assert local_rank in [0, 1]
    if rank in [2, 3]: # Ranks in the second TP group
        local_rank = dist.get_rank(group=tp_group)
        assert local_rank in [0, 1]


@pytest.mark.world_size(8)
def test_3d_mesh_coordinates(distributed_env):
    """
    Tests the coordinate calculation for a 2x2x2 3D mesh.

    This test requires a world size of 8.
    - Mesh: (dp=2, pp=2, tp=2)
    - Global ranks:
      tensor([[[0, 1],   # dp=0, pp=0, tp=0-1
               [2, 3]],  # dp=0, pp=1, tp=0-1

              [[4, 5],   # dp=1, pp=0, tp=0-1
               [6, 7]]]) # dp=1, pp=1, tp=0-1
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size != 8:
        pytest.skip("This test requires a world size of 8.")

    mesh_dim = (2, 2, 2)
    mesh_name = ('dp', 'pp', 'tp')

    # Create a mesh tensor representing the global ranks in the 2x2x2 grid
    mesh_tensor = torch.arange(world_size).view(mesh_dim)
    # Initialize the MeshGenerator
    device_mesh = MeshGenerator(mesh_tensor, 'cuda', mesh_dim, mesh_name, is_backend_initialised=True)

    # Define the expected coordinates for each global rank
    expected_coords = {
        0: [0, 0, 0], 1: [0, 0, 1],
        2: [0, 1, 0], 3: [0, 1, 1],
        4: [1, 0, 0], 5: [1, 0, 1],
        6: [1, 1, 0], 7: [1, 1, 1],
    }

    # Get coordinates for the current rank and assert correctness
    coords = device_mesh.get_coordinates_tensor_search(rank)
    assert coords == expected_coords[rank]

    # Additionally, test a specific group (e.g., PP group) to ensure it's accessible
    pp_group = device_mesh.get_group('pp')
    # Ranks 0,1,4,5 are in PP groups where pp_coord is 0.
    # Ranks 2,3,6,7 are in PP groups where pp_coord is 1.
    # For example, ranks 0 and 2 form a PP group (dp=0, tp=0).
    # Ranks 4 and 6 form a PP group (dp=1, tp=0).
    if rank in [0, 2, 4, 6]: # Ranks that are the first in their respective PP groups
        local_rank = dist.get_rank(group=pp_group)
        assert local_rank == 0
    if rank in [1, 3, 5, 7]: # Ranks that are the second in their respective PP groups
        local_rank = dist.get_rank(group=pp_group)
        assert local_rank == 1
