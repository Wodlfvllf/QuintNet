"""
Tensor parallel training implementation
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
import sys

# Import from utilities package  
from ..utilities.utils import *
from ..utilities.Dataloader import CustomDataset, mnist_transform
from ..utilities.model import Attention, Model, PatchEmbedding, MLP

# Import tensor parallelism components
from QuintNet.TensorParallelism import All_Gather, ColumnParallelLinear, apply_tensor_parallel, ProcessGroupManager



def train_epoch(model, train_loader, criterion, optimizer, device, rank):
    """Train model for one epoch with tensor parallelism support."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    debug_freq=50
    gradient_norms = []
    output_stats = []
    
    if rank == 0:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)        
        labels = batch['label'].to(device)
        
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
                'Loss': f'{running_loss/len(pbar):.4f}',
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
    
    # Since all ranks process the same data, divide by world_size
    world_size = dist.get_world_size()
    avg_loss = (total_loss / world_size).item() / len(train_loader)
    avg_accuracy = (total_correct / world_size).item() / (total_samples / world_size).item() * 100
    
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device, rank):
    """Validate model with tensor parallelism support."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
            
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if rank == 0:
                accuracy = 100 * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/len(pbar):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
    
    # Aggregate metrics across all ranks
    total_loss = torch.tensor(running_loss, device=device)
    total_correct = torch.tensor(correct, device=device)
    total_samples = torch.tensor(total, device=device)
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    # Since all ranks process the same data, divide by world_size
    world_size = dist.get_world_size()
    avg_loss = (total_loss / world_size).item() / len(val_loader)
    avg_accuracy = (total_correct / world_size).item() / (total_samples / world_size).item() * 100
    
    return avg_loss, avg_accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, rank):
    """Complete training loop with tensor parallelism support."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    patience=3
    min_improvement=0.01
    rollback_on_decrease=True
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    consecutive_decreases = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
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
        
        # Check for improvement
        improved = val_acc > best_val_acc + min_improvement
        decreased = len(val_accs) > 1 and val_acc < val_accs[-2]
        
        if improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            consecutive_decreases = 0
            
            # Save best model state (all ranks need this for potential rollback)
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }
            
            if rank == 0:
                torch.save(best_model_state, 'best_model.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
                
        else:
            epochs_without_improvement += 1
            
            if decreased:
                consecutive_decreases += 1
                if rank == 0:
                    print(f"⚠ Validation accuracy decreased: {val_accs[-2]:.2f}% → {val_acc:.2f}%")
            else:
                consecutive_decreases = 0
        
        # Rollback logic
        if rollback_on_decrease and consecutive_decreases >= 2 and best_model_state is not None:
            if rank == 0:
                print(f"🔄 Rolling back to best model (epoch {best_model_state['epoch']+1}, acc: {best_model_state['val_acc']:.2f}%)")
            
            # Restore best model state on all ranks
            model.load_state_dict(best_model_state['model_state_dict'])
            optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            
            # Reset counters
            consecutive_decreases = 0
            epochs_without_improvement = 0
            
            # Optionally reduce learning rate after rollback
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            
            if rank == 0:
                print(f"📉 Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f}")
        
        if rank == 0:
            status = ""
            if improved:
                status = " 🎉 NEW BEST!"
            elif decreased:
                status = f" 📉 (-{val_accs[-2] - val_acc:.2f}%)"
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}")
            print(f"Best Val Acc: {best_val_acc:.2f}%, Patience: {patience - epochs_without_improvement}")
    
    # Load best model at the end
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        if rank == 0:
            print(f"\n🏆 Training finished. Best model loaded (Val Acc: {best_val_acc:.2f}%)")

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


def main():
    # Set seeds for reproducible shuffling across all ranks
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Distributed init
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)   # pin process to its GPU
    
    # Configuration
    config = {
        'dataset_path': '/workspace/dataset/',
        'batch_size': 64,
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'num_workers': 4
    }
    
    # Device
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Using {world_size} GPUs for tensor parallelism")
        print(f"Rank {rank} using device: {device}")
    
    # Set seeds on all ranks
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
    
    # Worker init function - SAME seed for all workers across all ranks
    def worker_init_fn(worker_id):
        set_seed(42)  # Fixed: same seed for all workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,  # Fixed function
        sampler=None,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        sampler=None,
        persistent_workers=True
    )
    
    # Verify data consistency across ranks
    if rank == 0:
        print("Verifying data consistency across ranks...")
    
    first_batch = next(iter(train_loader))
    batch_sum = first_batch['image'].sum().item()
    
    gathered_sums = [None] * world_size
    dist.all_gather_object(gathered_sums, batch_sum)
    
    if rank == 0:
        if len(set(gathered_sums)) == 1:
            print("✓ All ranks have identical data")
        else:
            print("✗ WARNING: Data differs across ranks!")
            print(f"Batch sums: {gathered_sums}")
    
    # Create model
    model = Model(        
        img_size=28, 
        patch_size=4, 
        hidden_dim=64, 
        in_channels=1, 
        n_heads=4,
        depth=4
    ).to(device)
    
    # Apply tensor parallelism
    tp_size = world_size   # using all GPUs for tensor parallelism
    model = apply_tensor_parallel(model, tp_size, method_of_parallelism="column")
    
    # Force enable gradients on all parameters
    for param in model.parameters():
        param.requires_grad_(True)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
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
        print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()