"""
Pipeline Parallelism Utilities.

Helper functions for:
- Layer distribution
- Tensor shape calculation
- Stage identification
"""

from typing import List


def distribute_layers(total_layers: int, num_stages: int) -> List[int]:
    """
    TODO: Extract from pp_wrapper.py
    
    Distribute layers across pipeline stages.
    
    Args:
        total_layers: Total number of layers in model
        num_stages: Number of pipeline stages
        
    Returns:
        List of layer counts per stage
    """
    pass
