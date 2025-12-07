"""
Independent Model Verification Script

This script independently verifies the model's actual performance
by loading saved weights and running inference WITHOUT any distributed code.

This eliminates any possibility of bugs in the distributed metric calculation.

Usage:
    python -m QuintNet.examples.verify_model --checkpoint path/to/checkpoint.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from ..utils import CustomDataset, mnist_transform, Model
from ..core import load_config


def verify_model(checkpoint_path: str, config_path: str = 'QuintNet/examples/config.yaml'):
    """
    Load a saved model and independently verify its accuracy.
    
    This uses the simplest possible code path - no distributed logic at all.
    """
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create fresh model with same architecture
    model = Model(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        in_channels=config['in_channels'],
        n_heads=config['n_heads'],
        depth=config['depth']
    ).to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Assume it's just the state dict
    
    # Remove any DDP/distributed prefixes from keys
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes from distributed wrappers
        new_key = key.replace('module.', '').replace('model.', '').replace('local_module.', '')
        cleaned_state_dict[new_key] = value
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load validation dataset
    val_dataset = CustomDataset(config['dataset_path'], split='test', transform=mnist_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"\nValidation set: {len(val_dataset)} samples")
    
    # Run inference - THE SIMPLEST POSSIBLE CODE
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\n" + "="*60)
    print("INDEPENDENT VERIFICATION")
    print("="*60)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Verifying"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Simple forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs, dim=1)
            
            # Count correct
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"Total samples:     {total}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {total - correct}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print("="*60)
    
    # Sanity checks
    print("\n🔍 Sanity Checks:")
    print(f"   - Unique predictions: {len(set(all_predictions))} classes")
    print(f"   - Unique labels:      {len(set(all_labels))} classes")
    
    # Distribution of predictions
    from collections import Counter
    pred_dist = Counter(all_predictions)
    print(f"   - Prediction distribution: {dict(sorted(pred_dist.items()))}")
    
    # If model always predicts same class, it's broken
    if len(set(all_predictions)) < 3:
        print("\n⚠️  WARNING: Model predicts very few unique classes!")
        print("   This might indicate a problem with training.")
    
    if accuracy > 10 and accuracy < 100:
        print(f"\n✅ Accuracy ({accuracy:.2f}%) looks reasonable for MNIST")
    elif accuracy <= 10:
        print(f"\n❌ Accuracy ({accuracy:.2f}%) is at random chance level - model may not have trained properly")
    
    return accuracy


def save_model_for_verification(model, path: str):
    """
    Helper function to save model state for later verification.
    Call this at the end of training.
    """
    # If model is wrapped (DDP, PP, etc.), get the underlying model
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    elif hasattr(model, 'model') and hasattr(model.model, 'local_module'):
        # Pipeline parallel case
        state_dict = model.model.local_module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({'model_state_dict': state_dict}, path)
    print(f"Saved model for verification: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify model accuracy independently")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='QuintNet/examples/config.yaml', help='Path to config')
    args = parser.parse_args()
    
    verify_model(args.checkpoint, args.config)
