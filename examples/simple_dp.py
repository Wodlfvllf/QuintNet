"""
Simple Data Parallel (DP) training example using the QuintNet Trainer.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import time

from QuintNet.utils.Dataloader import CustomDataset, mnist_transform
from QuintNet.utils.model import Model
from QuintNet.core.config import load_config
from QuintNet.core.process_groups import init_process_groups
from QuintNet.strategy import get_strategy
from QuintNet.trainer import Trainer

def main():
    """
    Main function to set up and run the DP training process.
    """
    parser = argparse.ArgumentParser(description="QuintNet Data Parallel Training")
    parser.add_argument('--config', type=str, default='examples/dp_config.yaml',
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
    coords = pg_manager.get_coordinates_tensor_search(global_rank)
    dp_rank = coords[config['mesh_name'].index('dp')]
    dp_size = config['mesh_dim'][config['mesh_name'].index('dp')]
    
    train_dataset = CustomDataset(config['dataset_path'], split='train', transform=mnist_transform)
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=config['num_workers'], drop_last=False, pin_memory=True)

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
        print(f"Total model parameters: {total_params:,}\n")

    # Get and apply the 'dp' strategy
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
