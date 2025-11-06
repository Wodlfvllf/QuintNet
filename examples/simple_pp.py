
"""
Pipeline Parallel Training for MNIST Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import random
import os


# Import from utilities
from QuintNet.utils.utils import *
from QuintNet.utils.Dataloader import CustomDataset, mnist_transform
from QuintNet.utils.model import Model

# Import pipeline parallelism components
from QuintNet.parallelism.pipeline_parallel import PipelineParallelWrapper, PipelineTrainer
from QuintNet.core import init_mesh



class PipelineDataLoader:
    """
    Wrapper for DataLoader to support gradient accumulation.
    """
    def __init__(self, dataloader, grad_acc_steps):
        self.dataloader = dataloader
        self.grad_acc_steps = grad_acc_steps
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        # Convert to expected format
        return {
            "images": batch['image'],
            "labels": batch['label']
        }
    
    def __len__(self):
        return len(self.dataloader)


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize_model_weights(model, rank, pp_group):
    """Broadcast model weights from rank 0 to all ranks."""
    if rank == 0:
        print("Synchronizing model weights across all ranks...")
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=pp_group)
    
    dist.barrier(group=pp_group)
    if rank == 0:
        print("✓ Model weights synchronized\n")


def train_epoch(pipeline_trainer, pipeline_loader, tensor_shapes, device, dtype, rank, pp_size, epoch, schedule):
    """Train one epoch with accuracy tracking."""
    pipeline_trainer.model.train()
    
    num_batches = len(pipeline_loader) // pipeline_loader.grad_acc_steps
    total_loss = 0.0
    total_acc = 0.0
    
    # Progress bar only on last rank
    if rank == pp_size - 1:
        pbar = tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}")
    
    for step in range(num_batches):
        # Choose training schedule
        if schedule == "afab":
            loss, acc = pipeline_trainer.train_step_afab(
                pipeline_loader, tensor_shapes, device, dtype
            )
        else:  # 1f1b
            loss, acc = pipeline_trainer.train_step_1f1b(
                pipeline_loader, tensor_shapes, device, dtype
            )
        
        # Accumulate metrics (only on last rank)
        if rank == pp_size - 1:
            if loss is not None:
                total_loss += loss
                total_acc += acc
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
            pbar.update(1)
    
    if rank == pp_size - 1:
        pbar.close()
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc
    
    return None, None


def validate(pipeline_trainer, val_loader, tensor_shapes, device, dtype, rank, pp_size):
    """Validate the model with accuracy tracking."""
    pipeline_trainer.model.eval()
    
    val_loss, val_acc = pipeline_trainer.evaluate(
        val_loader, 
        tensor_shapes, 
        device, 
        dtype
    )
    
    return val_loss, val_acc


def train_model(config, device_mesh):
    """Main training function with accuracy tracking."""
    rank = dist.get_rank()
    coords = device_mesh.get_coordinates_tensor_search(rank)
    pp_rank = coords[1]
    pp_size = device_mesh.mesh_dim[1]
    pp_group = device_mesh.get_group('pp')
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print("\n" + "="*60)
        print("PIPELINE PARALLEL TRAINING")
        print("="*60)
        print(f"Pipeline stages: {pp_size}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Gradient accumulation steps: {config['grad_acc_steps']}")
        print(f"Schedule: {config['schedule'].upper()}")
        print("="*60 + "\n")
    
    # Set seeds
    set_seed(42)
    
    # Load datasets
    train_dataset = CustomDataset(
        config['dataset_path'],
        split='train',
        transform=mnist_transform
    )
    val_dataset = CustomDataset(
        config['dataset_path'],
        split='test',
        transform=mnist_transform
    )
    
    # Create dataloaders
    def worker_init_fn(worker_id):
        set_seed(42 + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,
        persistent_workers=True if config['num_workers'] > 0 else False,
        generator=torch.Generator().manual_seed(42),
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
    
    # Create model
    model = Model(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        in_channels=config['in_channels'],
        n_heads=config['n_heads'],
        depth=config['depth']
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}\n")
    
    # Synchronize weights
    synchronize_model_weights(model, rank, pp_group)
    
    # Create pipeline wrapper
    pp_model = PipelineParallelWrapper(model, device_mesh, pp_rank, pp_group, pp_size, device).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(pp_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create pipeline trainer
    pipeline_trainer = PipelineTrainer(
        pp_model,
        pp_group,
        criterion,
        device,
        optimizer=optimizer,
        max_grad_norm=config['max_grad_norm']
    )
    
    # Create pipeline dataloader
    pipeline_train_loader = PipelineDataLoader(train_loader, config['grad_acc_steps'])
    
    # Calculate tensor shapes for communication
    num_patches = (config['img_size'] // config['patch_size']) ** 2
    tensor_shapes = (config['batch_size'], num_patches + 1, config['hidden_dim'])
    dtype = torch.float32
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(config['num_epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            pipeline_trainer,
            pipeline_train_loader,
            tensor_shapes,
            device,
            dtype,
            rank,
            pp_size,
            epoch,
            config['schedule']
        )
        
        # Validate
        val_loss, val_acc = validate(
            pipeline_trainer,
            val_loader,
            tensor_shapes,
            device,
            dtype,
            rank,
            pp_size
        )
        
        # Print metrics and track best model
        if rank == pp_size - 1:
            # Store metrics
            if train_loss is not None:
                train_losses.append(train_loss)
                train_accs.append(train_acc)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            # Print metrics
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check for improvement
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print(f"  🎉 New Best Validation Accuracy!")
            else:
                epochs_without_improvement += 1
            
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            print(f"  Patience: {config['patience'] - epochs_without_improvement}/{config['patience']}")
            print(f"{'='*50}")
            
            # Early stopping
            if epochs_without_improvement >= config['patience']:
                print(f"\n⚠ Early stopping triggered")
                break
        
        dist.barrier(group=pp_group)
    
    # Print final results
    if rank == pp_size - 1:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
        print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    return None



def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    global_rank = dist.get_rank()
    torch.cuda.set_device(global_rank)
    

    # Create mesh for communication
    world_size = dist.get_world_size()
    device_mesh = init_mesh(mesh_dim=(1, world_size, 1), mesh_name=('dp', 'pp', 'tp'))
    
    # Configuration
    config = {
        'dataset_path': os.environ.get('DATASET_PATH', '/mnt/dataset/mnist/'),
        'batch_size': 8,
        'num_workers': 2,
        'img_size': 28,
        'patch_size': 4,
        'hidden_dim': 64,
        'in_channels': 1,
        'n_heads': 4,
        'depth': 8,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'grad_acc_steps': 4,
        'max_grad_norm': 1.0,
        'patience': 5,
        'schedule': os.environ.get('SCHEDULE', '1f1b'),
    }
    
    # Train
    start_time = time.time()
    train_model(config, device_mesh)
    
    if dist.get_rank() == 0:
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    
    # Cleanup
    dist.barrier(group=device_mesh.get_group('pp'))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
