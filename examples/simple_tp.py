"""
================================================================================
Simple Tensor Parallel (TP) Training Example
================================================================================

This example demonstrates pure Tensor Parallelism using QuintNet.

Tensor Parallelism splits individual layers across multiple GPUs. For Linear
layers, the weight matrix is sharded either by columns (ColumnParallelLinear)
or by rows (RowParallelLinear).

Key Concepts:
- Individual layer weights are split across GPUs
- Each GPU computes a portion of the layer output
- Results are combined using AllGather or AllReduce
- Useful for very large layers (e.g., LLM attention/FFN)

    Single Linear Layer Split Across 2 GPUs:
    
    Input: [batch, in_features]
           │
           ├────────────────────────────────────┤
           │                                    │
           ▼                                    ▼
    ┌──────────────┐                  ┌──────────────┐
    │   GPU 0      │                  │   GPU 1      │
    │ W[:, 0:N/2]  │                  │ W[:, N/2:N]  │
    │ (half cols)  │                  │ (half cols)  │
    └──────────────┘                  └──────────────┘
           │                                    │
           ▼                                    ▼
    Out: [batch, N/2]                 Out: [batch, N/2]
           │                                    │
           └────────────AllGather───────────────┘
                          │
                          ▼
                  Out: [batch, N]

Usage:
    torchrun --nproc_per_node=2 -m QuintNet.examples.simple_tp

Configuration:
    Uses tp_config.yaml with mesh_dim=[2] and mesh_name=['tp']
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse
import time
import os

from ..utils import CustomDataset, mnist_transform, Model
from ..core import load_config, init_process_groups
from ..strategy import get_strategy
from ..trainer import Trainer


def main():
    """
    Main function to set up and run Tensor Parallel training.
    """
    parser = argparse.ArgumentParser(description="QuintNet Tensor Parallel Training")
    parser.add_argument('--config', type=str, default='QuintNet/examples/tp_config.yaml',
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
        print("TENSOR PARALLEL TRAINING")
        print("=" * 60)
        print(f"Tensor parallel size: {world_size}")
        print(f"Strategy: Pure Tensor Parallelism")
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
    # STEP 3: Create DataLoaders
    # =========================================================================
    # In Tensor Parallelism, all GPUs process the SAME data.
    # Each GPU has a shard of the weights and computes a partial result.
    # No DistributedSampler needed - all ranks see the same batches.
    
    train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    if global_rank == 0:
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"Batch size: {config['batch_size']} (same on all GPUs)")
        print(f"Note: All GPUs process identical batches in TP")

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
        print(f"Total model parameters: {total_params:,}")
        # In TP, each GPU holds a fraction of the Linear layer weights
        # but full copies of other layers (LayerNorm, Embedding, etc.)
        print(f"Linear layer params per GPU: ~{total_params // world_size:,} (sharded)\n")

    # =========================================================================
    # STEP 5: Apply Tensor Parallel Strategy
    # =========================================================================
    # This will:
    # 1. Find all nn.Linear layers in the model
    # 2. Replace them with ColumnParallelLinear or RowParallelLinear
    # 3. Shard the weights across the TP group
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
