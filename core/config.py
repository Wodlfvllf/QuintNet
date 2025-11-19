"""
Configuration Management for QuintNet.

This module will contain:
- Configuration dataclasses
- YAML/JSON config loading
- Environment variable parsing
- Default configurations

Migration Source: Configuration logic scattered across training scripts
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import yaml
import os


@dataclass
class ParallelismConfig:
    """Configuration for parallelism strategies."""
    
    # Mesh dimensions
    mesh_dim: Tuple[int, int, int] = (1, 1, 1)  # (DP, TP, PP)
    mesh_names: Tuple[str, str, str] = ('dp', 'tp', 'pp')
    
    # Backend
    backend: str = 'nccl'
    device_type: str = 'cuda'
    
    # Timeout
    timeout_seconds: int = 1800


@dataclass
class TrainingConfig:
    """Base training configuration."""
    
    # Model
    model_name: str = 'vit'
    hidden_dim: int = 64
    
    # Training
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    
    # Optimization
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_acc_steps: int = 1
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_freq: int = 10
    
    # Logging
    log_freq: int = 10
    wandb_project: Optional[str] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): The path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
            
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO: Implement config merging
    
    Merge multiple configuration dictionaries.
    """
    pass
