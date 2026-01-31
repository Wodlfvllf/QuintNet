"""
Custom DataLoader Utilities

This module provides a `CustomDataset` class for loading HuggingFace datasets
and utility functions for common data transformations. It aims to simplify
the process of preparing data for training in PyTorch, especially when working
with datasets from the HuggingFace `datasets` library.

===============================================================================
CONCEPTUAL OVERVIEW:
===============================================================================

Data loading is a critical first step in any machine learning pipeline.
This module offers:

-   **`CustomDataset`**: A `torch.utils.data.Dataset` compatible class that
    can load datasets stored in HuggingFace's `datasets` format (either as
    directories or Arrow files). It handles splitting and provides a standard
    `__len__` and `__getitem__` interface.
-   **Transformation Functions**: Helper functions like `pil_to_tensor_transform`
    and `mnist_transform` to convert image data into PyTorch tensors and apply
    common preprocessing steps (e.g., normalization).

This modular approach allows for easy integration of various datasets and
customizable preprocessing pipelines.

===============================================================================
"""

from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union, Tuple, List
import pandas as pd

class CustomDataset(TorchDataset):
    """
    A PyTorch-compatible dataset class that loads HuggingFace datasets.

    Supports loading datasets from both HuggingFace dataset directories
    (which can contain multiple splits) and single Arrow files.
    """
    
    def __init__(self, dataset_path: str, split: str = "train", transform: Optional[Callable] = None):
        """
        Initializes the CustomDataset.

        Args:
            dataset_path (str): Path to the dataset. This can be a directory
                containing a HuggingFace dataset (e.g., downloaded via `load_dataset`)
                or a single `.arrow` file.
            split (str): The dataset split to load (e.g., "train", "validation", "test").
                Only applicable if `dataset_path` points to a directory with multiple splits.
            transform (Optional[Callable]): An optional transform function to be
                applied to each sample retrieved by `__getitem__`.
        
        Raises:
            FileNotFoundError: If the `dataset_path` does not exist.
            ValueError: If the file type is unsupported or the specified split
                is not found in a multi-split dataset.
        """
        self.split = split
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> Dataset:
        """
        Internal method to load the dataset based on the path type.

        Returns:
            datasets.Dataset: The loaded HuggingFace dataset object.
        
        Raises:
            ValueError: If the file type is unsupported.
        """
        if self.dataset_path.is_dir():
            return self._load_from_directory()
        elif self.dataset_path.suffix == '.arrow':
            # Load from a single Arrow file
            return Dataset.from_file(str(self.dataset_path))
        else:
            raise ValueError(f"Unsupported file type: {self.dataset_path}. Expected a directory or a .arrow file.")
    
    def _load_from_directory(self) -> Dataset:
        """
        Internal method to load a dataset from a HuggingFace dataset directory.

        Handles `DatasetDict` (multiple splits) or single `Dataset` objects.

        Returns:
            datasets.Dataset: The loaded HuggingFace dataset object for the specified split.
        
        Raises:
            ValueError: If the specified split is not found in a `DatasetDict`.
        """
        dataset = load_from_disk(str(self.dataset_path))
        
        if isinstance(dataset, DatasetDict):
            if self.split not in dataset:
                available_splits = list(dataset.keys())
                raise ValueError(f"Split '{self.split}' not found in dataset. Available splits: {available_splits}")
            return dataset[self.split]
        else:
            # If the directory contains a single dataset without explicit splits
            return dataset
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: The sample, potentially with transformations applied.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if idx >= len(self.dataset) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset)}")
        
        sample = self.dataset[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def pil_to_tensor_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    A transformation function that converts PIL Image objects within a sample
    dictionary to PyTorch tensors.

    This transform is typically used for image datasets.

    Args:
        sample (Dict[str, Any]): A dictionary representing a single data sample,
            which may contain PIL Image objects.

    Returns:
        Dict[str, Any]: The sample dictionary with PIL Images converted to
            PyTorch tensors.
    """
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.ToTensor()
        
        # Iterate through the sample and apply the transform to any PIL Image instances
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                sample[key] = transform(value)
                
    except ImportError:
        # If torch or torchvision is not available, skip the transform
        print("Warning: PyTorch or torchvision not found. Skipping PIL to Tensor transform.")
        pass
    
    return sample


def mnist_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    A specific transformation function for MNIST dataset samples.

    It converts PIL images to PyTorch tensors and applies standard MNIST
    normalization. It also converts the label to a `torch.long` tensor.

    Args:
        sample (Dict[str, Any]): A dictionary representing a single MNIST sample,
            expected to contain 'image' (PIL Image) and 'label' (int).

    Returns:
        Dict[str, Any]: The transformed MNIST sample.
    """
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
        ])
        
        if 'image' in sample and isinstance(sample['image'], Image.Image):
            sample['image'] = transform(sample['image'])
        
        if 'label' in sample:
            sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
            
    except ImportError:
        # If torch or torchvision is not available, skip the transform
        print("Warning: PyTorch or torchvision not found. Skipping MNIST transform.")
        pass
    
    return sample

class SummarizationDataset(TorchDataset):
    """
    Dataset for text summarization (e.g., CNN/DailyMail).
    
    Loads CSV files with 'article' and 'highlights' columns.
    Returns raw text - tokenization happens in the collator.
    
    Args:
        dataset_path: Path to directory containing train.csv, validation.csv, test.csv
        split: Which split to load ('train', 'validation', 'test')
    """
    
    def __init__(self, dataset_path: Union[str, Path], split: str = 'train'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.data = self._load_dataset()
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load the appropriate CSV file based on split."""
        if self.split == 'train':
            csv_path = self.dataset_path / 'train.csv'
        elif self.split == 'validation':
            csv_path = self.dataset_path / 'validation.csv'
        elif self.split == 'test':
            csv_path = self.dataset_path / 'test.csv'
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'validation', or 'test'")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        return pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        row = self.data.iloc[idx]
        return {
            'article': str(row['article']),
            'highlights': str(row['highlights'])
        }


class SummarizationCollator:
    """
    Collator for causal language modeling (GPT-2 style).
    
    For summarization, we concatenate: article + separator + highlights
    The model learns to generate highlights given the article.
    
    Args:
        tokenizer: HuggingFace tokenizer (e.g., GPT2Tokenizer)
        max_length: Maximum sequence length
        max_target_length: Maximum length for highlights portion
    """
    
    def __init__(self, tokenizer, max_length: int = 512, max_target_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and prepare batch for CLM training.
        
        Format: <article> TL;DR: <highlights><eos>
        Labels: -100 for article portion, token ids for highlights portion
        """
        input_texts = []
        for sample in batch:
            # Create prompt format for summarization
            text = f"{sample['article']}\n\nTL;DR: {sample['highlights']}{self.tokenizer.eos_token}"
            input_texts.append(text)
        
        # Tokenize all texts
        encodings = self.tokenizer(
            input_texts,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # For CLM, labels = input_ids shifted by 1 (handled by loss function)
        # We mask padding tokens with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class SummarizationDataLoader:
    """
    Convenience wrapper for DataLoader with SummarizationDataset.
    
    Args:
        dataset: SummarizationDataset instance
        batch_size: Batch size
        collator: SummarizationCollator instance
        shuffle: Whether to shuffle (default True for training)
        num_workers: Number of dataloader workers
    """
    
    def __init__(
        self, 
        dataset: SummarizationDataset, 
        batch_size: int, 
        collator: SummarizationCollator,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collator = collator
        self.data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collator,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def __len__(self) -> int:
        return len(self.data_loader)
    
    def __iter__(self):
        return iter(self.data_loader)
