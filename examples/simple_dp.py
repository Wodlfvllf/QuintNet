"""
================================================================================
Simple Data Parallel (DP) Training Example
================================================================================

This example demonstrates pure Data Parallelism using QuintNet.

Data Parallelism replicates the entire model on each GPU. Each GPU processes
a different subset of the data (mini-batch), and gradients are synchronized
across all GPUs after each backward pass using AllReduce.

Key Concepts:
- Model is replicated on all GPUs
- Data is split across GPUs using DistributedSampler
- Gradients are averaged across all replicas
- All GPUs have identical model weights after each step

Usage:
    torchrun --nproc_per_node=4 -m QuintNet.examples.simple_dp

Configuration:
    Uses dp_config.yaml with mesh_dim=[4] and mesh_name=['dp']
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import time
import os

from ..utils import CustomDataset, mnist_transform, Model
from ..core import load_config, init_process_groups
from ..strategy import get_strategy
from ..trainer import Trainer


def main():
    """
    Main function to set up and run Data Parallel training.
    """
    parser = argparse.ArgumentParser(description="QuintNet Data Parallel Training")
    parser.add_argument('--config', type=str, default='QuintNet/examples/dp_config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # =========================================================================
    # STEP 1: Initialize Distributed Environment
    # =========================================================================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("DATA PARALLEL TRAINING")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Strategy: Pure Data Parallelism")
        print("=" * 60 + "\n")

    # =========================================================================
    # STEP 2: Initialize Process Group Manager
    # =========================================================================
    pg_manager = init_process_groups(
        device_type=config['device_type'],
        mesh_dim=config['mesh_dim'],
        mesh_name=config['mesh_name']
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    # =========================================================================
    # STEP 3: Create DataLoaders with DistributedSampler
    # =========================================================================
    # In Data Parallelism, each GPU gets a different subset of the data
    dp_rank = global_rank  # For pure DP, global_rank == dp_rank
    dp_size = world_size   # For pure DP, world_size == dp_size
    
    train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    
    # DistributedSampler ensures each GPU gets unique data
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dp_size, 
        rank=dp_rank, 
        shuffle=True, 
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=dp_size, 
        rank=dp_rank, 
        shuffle=False, 
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=train_sampler, 
        num_workers=config['num_workers'], 
        drop_last=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        sampler=val_sampler, 
        num_workers=config['num_workers'], 
        drop_last=False, 
        pin_memory=True
    )

    if global_rank == 0:
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"Batch size per GPU: {config['batch_size']}")
        print(f"Effective batch size: {config['batch_size'] * dp_size}")

    # =========================================================================
    # STEP 4: Create Model
    # =========================================================================
    model = Model(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        in_channels=config['in_channels'],
        n_heads=config['n_heads'],
        depth=config['depth']
    )
    
    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Parameters per GPU: {total_params:,} (full model replicated)\n")

    # =========================================================================
    # STEP 5: Apply Data Parallel Strategy
    # =========================================================================
    strategy = get_strategy(config['strategy_name'], pg_manager, config)
    parallel_model = strategy.apply(model)

    # =========================================================================
    # STEP 6: Train!
    # =========================================================================
    trainer = Trainer(parallel_model, train_loader, val_loader, config, pg_manager)
    
    start_time = time.time()
    trainer.fit()
    
    if global_rank == 0:
        elapsed_time = time.time() - start_time
        print(f"\nTotal training time: {elapsed_time:.2f} seconds")
        print(f"Average time per epoch: {elapsed_time / config['num_epochs']:.2f} seconds")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
