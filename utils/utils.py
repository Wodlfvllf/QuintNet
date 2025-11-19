"""
General Utility Functions for QuintNet

This module provides a collection of general-purpose utility functions that
support various aspects of the QuintNet framework, including device setup,
model inspection, and checkpoint management.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

These utilities are designed to simplify common tasks in deep learning
development, making the main training scripts cleaner and more focused
on the core logic.

Key functionalities include:
-   **Device Setup**: Automatically determines and sets up the appropriate
    computing device (GPU or CPU).
-   **Model Inspection**: Provides a function to easily count the number of
    trainable parameters in a model.
-   **Checkpointing**: Offers standardized functions for saving and loading
    model and optimizer states, crucial for resuming training or deploying models.

===============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# from torchvision.transforms import v2 # Not directly used in functions, can be removed if not needed elsewhere
# from einops import rearrange # Not directly used in functions, can be removed if not needed elsewhere
# import numpy as np # Not directly used in functions, can be removed if not needed elsewhere
# import torch.nn.functional as F # Not directly used in functions, can be removed if not needed elsewhere

def setup_device() -> torch.device:
    """
    Sets up and returns the appropriate computing device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str):
    """
    Saves a model checkpoint to the specified path.

    The checkpoint includes the current epoch, model state dictionary,
    optimizer state dictionary, and the current loss.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current training loss.
        path (str): The file path where the checkpoint will be saved.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    """
    Loads a model checkpoint from the specified path.

    Args:
        model (nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer instance to load the state into.
        path (str): The file path of the checkpoint to load.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, int, float]: A tuple containing
            the loaded model, optimizer, epoch number, and loss.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss