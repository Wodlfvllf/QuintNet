"""
Integrated training with 3D Parallelism (DP + PP + TP).

Supports two modes:
1. Default mode: Train a ViT model on MNIST (original behavior)
2. Staged mode: Load GPT-2 from checkpoint with distributed loading

Usage:
    # Default ViT on MNIST:
    torchrun --nproc_per_node=8 -m QuintNet.examples.full_3d --config examples/config.yaml

    # GPT-2 with distributed loading:
    torchrun --nproc_per_node=8 -m QuintNet.examples.full_3d \\
        --checkpoint /path/to/gpt2.safetensors \\
        --staged \\
        --model_config /path/to/gpt2_config.json
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import time
import os
from torch.utils.data.distributed import DistributedSampler
import argparse

from ..utils import CustomDataset, mnist_transform, Model
from ..core import load_config, init_process_groups
from ..strategy import get_strategy
from ..trainer import Trainer


def main():
    """
    Main function to set up and run the training process.
    """
    parser = argparse.ArgumentParser(description="QuintNet 3D Parallelism Training")
    parser.add_argument('--config', type=str, default='examples/config.yaml',
                        help='Path to the YAML configuration file.')
    
    # ─────────────────────────────────────────────────────────────────
    # NEW: Arguments for staged/distributed loading mode
    # ─────────────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to safetensors checkpoint for distributed loading.')
    parser.add_argument('--staged', action='store_true',
                        help='Enable staged mode: load weights from checkpoint with distributed loading.')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config JSON (for staged mode, e.g., GPT-2 config).')
    
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)
    
    # If model_config is provided, merge it into config
    if args.model_config:
        import json
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
        config['model_config'] = model_config

    # Initialize distributed environment
    # Explicitly set the device based on LOCAL_RANK to avoid NCCL hangs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    global_rank = dist.get_rank()

    # Initialize the ProcessGroupManager
    pg_manager = init_process_groups(
        device_type=config['device_type'],
        mesh_dim=config['mesh_dim'],
        mesh_name=config['mesh_name']
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Get coordinates for data loading
    coords = pg_manager.get_coordinates_tensor_search(global_rank)
    dp_rank = coords[config['mesh_name'].index('dp')]
    dp_size = config['mesh_dim'][config['mesh_name'].index('dp')]

    # ═══════════════════════════════════════════════════════════════════
    # MODE SELECTION: Staged (distributed loading) vs Default (ViT)
    # ═══════════════════════════════════════════════════════════════════
    
    if args.staged:
        # ─────────────────────────────────────────────────────────────────
        # STAGED MODE: Load GPT-2 from checkpoint
        # ─────────────────────────────────────────────────────────────────
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --staged is set")
        
        if global_rank == 0:
            print(f"[Staged Mode] Loading from checkpoint: {args.checkpoint}")
        
        # In staged mode, model is None - it will be built from checkpoint
        model = None
        
        # Get strategy with checkpoint path
        strategy = get_strategy(
            config['strategy_name'], 
            pg_manager, 
            config,
            checkpoint_path=args.checkpoint,
            is_staged=True,
        )
        parallel_model = strategy.apply(model)
        
        # TODO: Add GPT-2 specific dataset/dataloader here
        # For now, using placeholder
        train_loader = None
        val_loader = None
        
        if global_rank == 0:
            print("[Staged Mode] Model parallelized successfully!")
            print("Note: Dataset/trainer for GPT-2 not yet implemented")
        
    else:
        # ─────────────────────────────────────────────────────────────────
        # DEFAULT MODE: Train ViT on MNIST (original behavior)
        # ─────────────────────────────────────────────────────────────────
        
        # Create DataLoaders
        train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
        val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
        train_sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=42)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False, seed=42)
        
        # Calculate micro-batch size for pipeline parallelism
        micro_batch_size = config['batch_size'] // (config['grad_acc_steps'] * dp_size)
        if micro_batch_size < 1:
            micro_batch_size = 1
        
        if global_rank == 0:
            print(f"[Default Mode] Using micro_batch_size={micro_batch_size}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=micro_batch_size, 
            sampler=train_sampler, 
            num_workers=config['num_workers'], 
            drop_last=True, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=micro_batch_size, 
            sampler=val_sampler, 
            num_workers=config['num_workers'], 
            drop_last=False, 
            pin_memory=True
        )

        # Create the base model (ViT)
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
            print(f"Total model parameters: {total_params:,}\n")

        # Get and apply the parallelism strategy
        strategy = get_strategy(config['strategy_name'], pg_manager, config)
        parallel_model = strategy.apply(model)

        # Create and run the Trainer
        trainer = Trainer(parallel_model, train_loader, val_loader, config, pg_manager)
        
        start_time = time.time()
        trainer.fit()
        
        if global_rank == 0:
            elapsed_time = time.time() - start_time
            print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
