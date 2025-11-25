"""
Simple Tensor Parallel (TP) training example using the QuintNet Trainer.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse
import time

from ..utils import CustomDataset, mnist_transform, Model
from ..core import load_config, init_process_groups
from ..strategy import get_strategy
from ..trainer import Trainer

def main():
    """
    Main function to set up and run the TP training process.
    """
    parser = argparse.ArgumentParser(description="QuintNet Tensor Parallel Training")
    parser.add_argument('--config', type=str, default='examples/tp_config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()

    # Initialize the ProcessGroupManager
    pg_manager = init_process_groups(
        device_type=config['device_type'],
        mesh_dim=config['mesh_dim'],
        mesh_name=config['mesh_name']
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create DataLoaders
    # NOTE: For pure TP, there is no data distribution, so we don't need a DistributedSampler.
    # Each GPU processes the same data.
    train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Create the base model
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
        print(f"Total base model parameters: {total_params:,}\n")

    # Get and apply the 'tp' strategy
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
