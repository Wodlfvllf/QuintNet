"""
Memory Tracking and Management.
"""

import torch
from typing import Dict, Optional


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory stats (allocated, reserved, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def clear_cache():
    """
    Clear CUDA cache to free memory.
    """
    torch.cuda.empty_cache()
