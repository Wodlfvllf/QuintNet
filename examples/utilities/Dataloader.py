from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
import os
from pathlib import Path

class CustomDataset(TorchDataset):
    """
    A PyTorch-compatible dataset class that loads HuggingFace datasets.
    Supports both dataset directories and single Arrow files.
    """
    
    def __init__(self, dataset_path, split="train", transform=None):
        """
        Args:
            dataset_path: Path to dataset folder or .arrow file
            split: Dataset split to load ("train", "test", etc.)
            transform: Optional transform function for samples
        """
        self.split = split
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on path type."""
        if self.dataset_path.is_dir():
            return self._load_from_directory()
        elif self.dataset_path.suffix == '.arrow':
            return Dataset.from_file(str(self.dataset_path))
        else:
            raise ValueError(f"Unsupported file type: {self.dataset_path}")
    
    def _load_from_directory(self):
        """Load dataset from HuggingFace dataset directory."""
        dataset = load_from_disk(str(self.dataset_path))
        
        if isinstance(dataset, DatasetDict):
            if self.split not in dataset:
                available_splits = list(dataset.keys())
                raise ValueError(f"Split '{self.split}' not found. Available: {available_splits}")
            return dataset[self.split]
        else:
            return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx >= len(self.dataset) or idx < 0:
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.dataset[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def pil_to_tensor_transform(sample):
    """Transform that converts PIL images to PyTorch tensors."""
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.ToTensor()
        
        # Convert any PIL images in the sample
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                sample[key] = transform(value)
                
    except ImportError:
        pass  # Skip transform if PyTorch/torchvision not available
    
    return sample


def mnist_transform(sample):
    """MNIST-specific transform with normalization."""
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        if 'image' in sample and isinstance(sample['image'], Image.Image):
            sample['image'] = transform(sample['image'])
        
        if 'label' in sample:
            sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
            
    except ImportError:
        pass
    
    return sample


# # Usage examples
# if __name__ == "__main__":
#     # Basic usage
#     train_dataset = CustomDataset("/workspace/datasets/mnist", split="train", transform=pil_to_tensor_transform)
#     test_dataset = CustomDataset("/workspace/datasets/mnist", split="test", transform=pil_to_tensor_transform)
#     print(train_dataset.__getitem__(0))
#     # With PyTorch DataLoader
#     from torch.utils.data import DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)