"""
Configuration Management for QuintNet

This module provides utilities for loading and managing configurations for
training runs. It is designed to load settings from a YAML file, providing a
flexible way to manage hyperparameters and model configurations without
modifying the source code.

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

The primary function is `load_config`, which is used in the main training
script to load all settings from a YAML file.

.. code-block:: python

    # In config.yaml
    learning_rate: 1e-4
    model:
      name: 'vit'
      depth: 12

    # In the main training script
    from QuintNet.core.config import load_config

    config = load_config('path/to/config.yaml')
    lr = config['learning_rate']  # 1e-4
    model_depth = config['model']['depth'] # 12

===============================================================================
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import yaml
import os


@dataclass
class ParallelismConfig:
    """
    A dataclass for storing parallelism-specific configurations.
    
    Note: This is currently not used as settings are loaded directly from YAML,
    but it serves as a schema for the expected configuration structure.
    """
    # Mesh dimensions for (DP, TP, PP)
    mesh_dim: Tuple[int, int, int] = (1, 1, 1)
    mesh_names: Tuple[str, str, str] = ('dp', 'tp', 'pp')
    
    # Distributed backend settings
    backend: str = 'nccl'
    device_type: str = 'cuda'
    
    # Communication timeout
    timeout_seconds: int = 1800


@dataclass
class TrainingConfig:
    """
    A dataclass for storing training-specific configurations.

    Note: This is currently not used as settings are loaded directly from YAML,
    but it serves as a schema for the expected configuration structure.
    """
    # Model settings
    model_name: str = 'vit'
    hidden_dim: int = 64
    
    # Training loop settings
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    
    # Optimizer settings
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_acc_steps: int = 1
    
    # Hardware and data loading settings
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing settings
    save_dir: str = './checkpoints'
    save_freq: int = 10
    
    # Logging settings
    log_freq: int = 10
    wandb_project: Optional[str] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration from a specified YAML file path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration settings.
        
    Raises:
        FileNotFoundError: If the `config_path` does not exist.
        RuntimeError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            # Use safe_load to avoid arbitrary code execution
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
            
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO: Implement config merging
    
    Merges multiple configuration dictionaries. The last dictionary in the
    list has the highest precedence.
    """
    pass