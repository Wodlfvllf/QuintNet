import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Import from utilities package
from ..utils import CustomDataset, mnist_transform, Model
from ..utils.model import MLP
from ..core import load_config


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
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
        
        accuracy = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{running_loss/len(pbar):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return running_loss / len(train_loader), accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    return running_loss / len(val_loader), accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Complete training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
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
    # Configuration
    config = {
        'dataset_path': '/workspace/dataset/',
        'batch_size': 64,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'num_workers': 4
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    model = Model(        
        img_size = 28, 
        patch_size = 4, 
        hidden_dim = 64, 
        in_channels = 1, 
        n_heads = 4,
        depth = 4
        ).to(device)
    
    # Train
    start_time = time.time()
    results = train_model(
        model, 
        train_loader, 
        val_loader, 
        config['num_epochs'], 
        config['learning_rate'], 
        device
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()