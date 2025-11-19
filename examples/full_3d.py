"""
Integrated training of 3D Parallelism Training for MNIST Classification
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
import sys
from torch.utils.data.distributed import DistributedSampler
from ..tests import run_all_tests



from QuintNet.utils.utils import *
from QuintNet.utils.Dataloader import CustomDataset, mnist_transform
from QuintNet.utils.model import Model

# Import parallelism components
from QuintNet.core.process_groups import init_process_groups
from QuintNet.parallelism.hybrid.strategy import get_strategy
from QuintNet.parallelism.pipeline_parallel import PipelineTrainer




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


def synchronize_model_weights(model, pp_group):
    """Synchronize model weights across pipeline stages."""
    my_pp_rank = dist.get_rank(group=pp_group)

    if my_pp_rank == 0:
        print(f"Rank {dist.get_rank()}: Broadcasting weights to my pipeline group...")

    for param in model.parameters():
        # Broadcast from rank 0 of the pipeline group
        dist.broadcast(param.data, group_src=0, group=pp_group)

    dist.barrier(group=pp_group)


def train_epoch_without_pp(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device, 
    rank
):
    """Train model for one epoch with DDP support."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)        
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if rank == 0:
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    # Calculate average loss and accuracy for this rank
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_epoch_with_pp(
    pipeline_trainer, 
    pipeline_loader, 
    tensor_shapes, 
    device, 
    dtype, 
    rank, 
    pp_size, 
    epoch, 
    schedule
):
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


def run_pp_training_loop(
    config, 
    pipeline_trainer, 
    pipeline_train_loader, 
    val_loader, 
    tensor_shapes, 
    dtype, 
    device, 
    pp_rank, 
    pp_size, 
    pp_group
):
    """
    Executes the main training and validation loop over all epochs.
    Handles metric tracking, progress bars, and early stopping.
    """
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(config['num_epochs']):
        # Only the first rank of the first DP replica should print epoch info
        if pp_rank == 0 and dist.get_rank() < config['mesh_dim'][1] * config['mesh_dim'][2]:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}\n" + "-" * 50)
        
        # --- Train for one epoch ---
        train_loss, train_acc = train_epoch_with_pp(
            pipeline_trainer,
            pipeline_train_loader,
            tensor_shapes,
            device,
            dtype,
            pp_rank,
            pp_size,
            epoch,
            config['schedule']
        )
        
        # --- Validate after the epoch ---
        val_loss, val_acc = validate(
            pipeline_trainer,
            val_loader,
            tensor_shapes,
            device,
            dtype,
            pp_rank,
            pp_size
        )
        
        # --- Log metrics and check for early stopping (only on the last stage) ---
        if pp_rank == pp_size - 1:
            # Store metrics
            if train_loss is not None:
                train_losses.append(train_loss)
                train_accs.append(train_acc)
            
            if val_loss is not None:
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            # Print metrics
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1} Results (Rank {dist.get_rank()}):")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check for improvement
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print("  🎉 New Best Validation Accuracy!")
            else:
                epochs_without_improvement += 1
            
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            print(f"  Patience: {config['patience'] - epochs_without_improvement}/{config['patience']}")
            print(f"{'='*50}")
            
            # Early stopping check
            if epochs_without_improvement >= config['patience']:
                print("\n⚠ Early stopping triggered")
                break
        
        # Synchronize all processes in the pipeline before the next epoch
        dist.barrier(group=pp_group)
    
    # --- Final Results ---
    if pp_rank == pp_size - 1:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        if train_accs:
            print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
        if val_accs:
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


def train_model(config, device_mesh):
    """Main training function with accuracy tracking."""
    
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Get all the necessary info from the mesh
    dp_group = device_mesh.get_group('dp')
    tp_group = device_mesh.get_group('tp')
    pp_group = device_mesh.get_group('pp')

    coords = device_mesh.get_coordinates_tensor_search(global_rank)
    dp_rank = coords[0]  # Assuming mesh order is ('dp', 'tp', 'pp')
    tp_rank = coords[1]
    pp_rank = coords[2]
    pp_size = config['mesh_dim'][2]
    dp_size = config['mesh_dim'][0]
        
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
        worker_init_fn=worker_init_fn,
        persistent_workers=True if config['num_workers'] > 0 else False,
        generator=torch.Generator().manual_seed(42),
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
    
    # Create model
    model = Model(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        in_channels=config['in_channels'],
        n_heads=config['n_heads'],
        depth=config['depth']
    ).to(device)
    
    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}\n")
    
    # Synchronize weights across pipeline stages
    # synchronize_model_weights(model, pp_group)
    
    # Initialize Tensor parallelism
    tp_model = apply_tensor_parallel(
        model, 
        config['mesh_dim'][1], 
        tp_rank,
        tp_group,
        device,
        gather_output=True, 
        sync_gradients=True, 
        method_of_parallelism="column"
    )
    
    # Create pipeline wrapper
    pp_model = PipelineParallelWrapper(
        tp_model, 
        device_mesh,
        pp_rank,
        pp_group,
        config['mesh_dim'][2],
        device
    ).to(device)
    
    # Initialize DDP
    dp_model = CustomDDP(pp_model)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(dp_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    data_loader_for_training = train_loader
    
    if config['mesh_dim'][2] > 1:
        # Create pipeline trainer
        pipeline_trainer = PipelineTrainer(
            dp_model,
            device_mesh,
            pp_rank,
            pp_group,
            criterion,
            device,
            optimizer=optimizer,
            max_grad_norm=config['max_grad_norm']
        )
        # Create pipeline dataloader
        data_loader_for_training = PipelineDataLoader(train_loader, config['grad_acc_steps'])
        metrics = run_pp_training_loop(
            config,
            pipeline_trainer,
            data_loader_for_training,
            val_loader,
            pp_model.get_tensor_shapes(config['batch_size']),
            torch.float32,
            device,
            pp_rank,
            pp_size,
            pp_group
        )
    else:
        # Standard DDP training loop
        for epoch in range(config['num_epochs']):
            if global_rank == 0:
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}\n" + "-" * 50)
            
            train_loss, train_acc = train_epoch_without_pp(
                dp_model,
                train_loader,
                criterion,
                optimizer,
                device,
                global_rank
            )
            
            if global_rank == 0:
                print(f"\nEpoch {epoch+1} Results (Rank {global_rank}):")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"{'='*50}\n")
        
        metrics = None
        
    return metrics


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    global_rank = dist.get_rank()    
    
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
        'device_type': 'cuda',
        'mesh_dim': (2, 2, 2),
        'mesh_name': ('dp', 'tp', 'pp')
    }
    
    # Create mesh for effective communication
    device_mesh = init_mesh(
        device_type=config['device_type'],
        mesh_dim=config['mesh_dim'],
        mesh_name=config['mesh_name']
    )
    
    run_all_tests(device_mesh)
    
    # Train
    start_time = time.time()
    metrics = train_model(config, device_mesh)
    
    if global_rank == 0 and metrics is not None:
        elapsed_time = time.time() - start_time
        print(f"\nTotal training time: {elapsed_time:.2f} seconds")
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
