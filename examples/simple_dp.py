"""
Data parallel training implementation using PyTorch DDP
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import numpy as np
import random
import sys
import os
from datetime import timedelta

# Import from utilities package  
from QuintNet.utils.utils import *
from QuintNet.utils.Dataloader import CustomDataset, mnist_transform
from QuintNet.utils.model import Attention, Model, PatchEmbedding, MLP
from QuintNet.DataParallelsim import CustomDDP


def train_epoch(model, train_loader, criterion, optimizer, device, rank):
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


def validate(model, val_loader, criterion, device, rank):
    """Validate model with DDP support."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
            
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
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
    
    # Aggregate metrics across all ranks
    total_loss = torch.tensor(running_loss, device=device)
    total_correct = torch.tensor(correct, device=device)
    total_samples = torch.tensor(total, device=device)
    
    # All-reduce to get global metrics
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    # Calculate global averages
    world_size = dist.get_world_size()
    avg_loss = total_loss.item() / (len(val_loader) * world_size)
    avg_accuracy = total_correct.item() / total_samples.item() * 100
    
    return avg_loss, avg_accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, rank):
    """Complete training loop with DDP support."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)
    
    # Training parameters
    patience = 3
    min_improvement = 0.01
    rollback_on_decrease = True
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    consecutive_decreases = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler to ensure different shuffling each epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, rank)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Check for improvement (only on rank 0 to avoid state divergence)
        if rank == 0:
            improved = val_acc > best_val_acc + min_improvement
            decreased = len(val_accs) > 1 and val_acc < val_accs[-2]
            
            if improved:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                consecutive_decreases = 0
                
                # Save best model state
                best_model_state = {
                    'model_state_dict': model.model.state_dict(),  # Use .module for DDP
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }
                
                torch.save(best_model_state, 'best_model.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
                    
            else:
                epochs_without_improvement += 1
                
                if decreased:
                    consecutive_decreases += 1
                    print(f"⚠ Validation accuracy decreased: {val_accs[-2]:.2f}% → {val_acc:.2f}%")
                else:
                    consecutive_decreases = 0
            
            # Rollback logic
            should_rollback = (rollback_on_decrease and consecutive_decreases >= 2 and 
                             best_model_state is not None)
        else:
            # Other ranks don't make decisions, just set flags
            improved = False
            should_rollback = False
        
        # Broadcast rollback decision to all ranks
        rollback_tensor = torch.tensor(1 if should_rollback else 0, device=device)
        dist.broadcast(rollback_tensor, src=0)
        should_rollback = rollback_tensor.item() == 1
        
        if should_rollback:
            if rank == 0:
                print(f"🔄 Rolling back to best model (epoch {best_model_state['epoch']+1}, acc: {best_model_state['val_acc']:.2f}%)")
            
            # Load best model on rank 0 and broadcast to other ranks
            if rank == 0 and best_model_state is not None:
                model.model.load_state_dict(best_model_state['model_state_dict'])
                optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            
            # Synchronize model state across all ranks
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            
            # Reset counters (only on rank 0, but broadcast decisions)
            if rank == 0:
                consecutive_decreases = 0
                epochs_without_improvement = 0
                
                # Reduce learning rate after rollback
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                
                print(f"📉 Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f}")
        
        if rank == 0:
            status = ""
            if improved:
                status = " 🎉 NEW BEST!"
            elif len(val_accs) > 1 and val_acc < val_accs[-2]:
                status = f" 📉 (-{val_accs[-2] - val_acc:.2f}%)"
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}")
            print(f"Best Val Acc: {best_val_acc:.2f}%, Patience: {patience - epochs_without_improvement}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            if rank == 0:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model at the end (only on rank 0, then broadcast)
    if rank == 0 and best_model_state is not None:
        model.model.load_state_dict(best_model_state['model_state_dict'])
        print(f"\n🏆 Training finished. Best model loaded (Val Acc: {best_val_acc:.2f}%)")
    
    # Synchronize final model state across all ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc if rank == 0 else None
    }


def setup_ddp_v2(rank, world_size):
    """Solution 2: NCCL with proper environment variables"""
    print(f"Rank {rank}: Setting up DDP with NCCL + env vars...")
    
    print(f"Rank {rank}: Environment variables set")
    print(f"Rank {rank}: Initializing process group...")
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60)  # Longer timeout for NCCL
    )
    print(f"Rank {rank}: Process group initialized with NCCL")
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: CUDA device set to {rank}")


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Get distributed training parameters
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = 4  # Assuming 4 GPUs for this example
    
    print(f"Rank {rank}: Setting up DDP...")
    
    # Setup distributed training
    setup_ddp_v2(rank, world_size)
    
    ddp_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    
    # Configuration
    config = {
        'dataset_path': '/mnt/dataset/mnist/',
        'batch_size': 64,  # Per GPU batch size
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'num_workers': 4
    }
    
    # Device
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Using {world_size} GPUs for data parallelism")
        print(f"Rank {rank} using device: {device}")
        print(f"Global batch size: {config['batch_size'] * world_size}")
    
    # Set seeds for reproducibility
    set_seed(42)
    
    print(f"Rank {rank}: Datasets...")

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
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,  # Use distributed sampler instead of shuffle
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Rank {rank}: Datasets loaded.")
    
    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Samples per GPU - Train: {len(train_loader.dataset) // world_size}, Val: {len(val_loader.dataset) // world_size}")
    
    print(f"Rank {rank}: Model loading.")
    
    # Create model
    model = Model(        
        img_size=28, 
        patch_size=4, 
        hidden_dim=64, 
        in_channels=1,
        n_heads=4,
        depth=4
    ).to(device)
    
    print(f"Rank {rank}: Model loaded.")
    
    # Wrap model with DDP
    model = CustomDDP(model)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        print(f"Model replicated across {world_size} GPUs")
    
    print(f"Rank {rank}: Model wrapped up.")
    print(f"Rank {rank}: Starting training.")

    # Train
    start_time = time.time()
    results = train_model(
        model, 
        train_loader, 
        val_loader, 
        config['num_epochs'], 
        config['learning_rate'], 
        device,
        rank
    )
    
    training_time = time.time() - start_time
    
    if rank == 0:
        print(f"\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s")
        if results['best_val_acc'] is not None:
            print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()