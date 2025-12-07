"""
================================================================================
Simple Pipeline Parallel (PP) Training Example
================================================================================

This example demonstrates pure Pipeline Parallelism using QuintNet.

Pipeline Parallelism splits the model into sequential stages, with each stage
running on a different GPU. Data flows through the pipeline in micro-batches,
allowing for efficient utilization of all GPUs.

Key Concepts:
- Model is split into stages (e.g., first half on GPU0, second half on GPU1)
- Uses micro-batching to keep pipeline filled
- 1F1B (One Forward, One Backward) schedule minimizes memory usage
- Activations are communicated between stages via Send/Recv

            GPU 0           GPU 1           GPU 2           GPU 3
         ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
         │ Stage 0 │────▶│ Stage 1 │────▶│ Stage 2 │────▶│ Stage 3 │
         │(Embed + │     │(Blocks  │     │(Blocks  │     │(Blocks +│
         │ Blocks) │     │  2-3)   │     │  4-5)   │     │  Head)  │
         └─────────┘     └─────────┘     └─────────┘     └─────────┘

Usage:
    torchrun --nproc_per_node=4 -m QuintNet.examples.simple_pp

Configuration:
    Uses pp_config.yaml with mesh_dim=[4] and mesh_name=['pp']
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
    Main function to set up and run Pipeline Parallel training.
    """
    parser = argparse.ArgumentParser(description="QuintNet Pipeline Parallel Training")
    parser.add_argument('--config', type=str, default='QuintNet/examples/pp_config.yaml',
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
        print("PIPELINE PARALLEL TRAINING")
        print("=" * 60)
        print(f"Pipeline stages: {world_size}")
        print(f"Strategy: Pure Pipeline Parallelism")
        print(f"Schedule: 1F1B (One Forward, One Backward)")
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
    # In Pipeline Parallelism without DP, ALL ranks need the same data
    # because the first stage reads it and passes activations forward.
    # However, each rank still needs to call next() to stay in sync.
    
    train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    
    # For PP with gradient accumulation, we use micro-batches
    micro_batch_size = config['batch_size'] // config['grad_acc_steps']
    if micro_batch_size < 1:
        micro_batch_size = 1
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=micro_batch_size, 
        shuffle=True, 
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=micro_batch_size, 
        shuffle=False, 
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    if global_rank == 0:
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"Micro-batch size: {micro_batch_size}")
        print(f"Gradient accumulation steps: {config['grad_acc_steps']}")
        print(f"Effective batch size: {micro_batch_size * config['grad_acc_steps']}")

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
        print(f"Parameters per GPU: ~{total_params // world_size:,} (model split)\n")

    # =========================================================================
    # STEP 5: Apply Pipeline Parallel Strategy
    # =========================================================================
    # This will:
    # 1. Split the model into stages
    # 2. Assign layers to each GPU
    # 3. Set up communication between stages
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