"""
PCAM (PatchCamelyon) Tissue Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the PCAM (PatchCamelyon)
tissue classification task. The dataset contains binary classification (normal vs tumor)
with data stored in HDF5 format.

Dataset structure:
  <data_root>/
    - camelyonpatch_level_2_split_train_x.h5 (images)
    - camelyonpatch_level_2_split_train_y.h5 (labels)
    - camelyonpatch_level_2_split_valid_x.h5 (images)
    - camelyonpatch_level_2_split_valid_y.h5 (labels)
    - camelyonpatch_level_2_split_test_x.h5 (images)
    - camelyonpatch_level_2_split_test_y.h5 (labels)

Key characteristics:
  - Binary classification task (normal vs tumor).
  - Data is stored in HDF5 format.
  - Train/valid/test splits are pre-defined.
  - Images are automatically resized to 224x224 if needed.

Usage example:
  from PCAM_dataset import PCAMDataset, create_pcam_dataloaders
  
  dataset = PCAMDataset(data_root='/path/to/PCAM', split='train')
  train_loader, valid_loader, test_loader = create_pcam_dataloaders(
      data_root='/path/to/PCAM',
      batch_size=32,
      num_workers=4
  )
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Please install: pip install h5py")

# -------------------------------
# Dataset class
# -------------------------------
class PCAMDataset(Dataset):
    """
    PCAM tissue classification dataset with pre-defined train/valid/test splits.
    
    Returns image and label pairs for binary classification: normal (0) vs tumor (1)
    """
    
    def __init__(self, data_root, split='train'):
        """
        Initialize PCAM dataset.

        Args:
            data_root: str - Root directory containing HDF5 files
            split: str - 'train', 'valid', or 'test'
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for PCAM dataset. Install with: pip install h5py")
        
        self.data_root = data_root
        self.split = split.lower()
        
        # Class names (binary classification)
        self.classes = ['normal', 'tumor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Determine HDF5 file paths based on split
        if self.split == 'train':
            x_file = os.path.join(data_root, 'camelyonpatch_level_2_split_train_x.h5')
            y_file = os.path.join(data_root, 'camelyonpatch_level_2_split_train_y.h5')
        elif self.split == 'valid':
            x_file = os.path.join(data_root, 'camelyonpatch_level_2_split_valid_x.h5')
            y_file = os.path.join(data_root, 'camelyonpatch_level_2_split_valid_y.h5')
        elif self.split == 'test':
            x_file = os.path.join(data_root, 'camelyonpatch_level_2_split_test_x.h5')
            y_file = os.path.join(data_root, 'camelyonpatch_level_2_split_test_y.h5')
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'valid', or 'test'")
        
        if not os.path.exists(x_file):
            raise ValueError(f"Image file not found: {x_file}")
        if not os.path.exists(y_file):
            raise ValueError(f"Label file not found: {y_file}")
        
        # Open HDF5 files (keep them open for efficient access)
        self.x_file = h5py.File(x_file, 'r')
        self.y_file = h5py.File(y_file, 'r')
        
        # Get data keys (usually 'x' and 'y')
        x_keys = list(self.x_file.keys())
        y_keys = list(self.y_file.keys())
        
        if len(x_keys) == 0 or len(y_keys) == 0:
            raise ValueError(f"Empty HDF5 files: {x_file} or {y_file}")
        
        self.x_key = x_keys[0]
        self.y_key = y_keys[0]
        
        # Get dataset length
        self.length = len(self.x_file[self.x_key])
        y_length = len(self.y_file[self.y_key])
        
        if self.length != y_length:
            raise ValueError(f"Mismatch in dataset lengths: images={self.length}, labels={y_length}")
        
        print(f"PCAM {self.split.upper()} set: {self.length} samples")
        print(f"Image shape: {self.x_file[self.x_key].shape}")
        print(f"Label shape: {self.y_file[self.y_key].shape}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """
        Get class distribution in current split.

        Returns:
            dict: Dictionary mapping class names to sample counts
        """
        from collections import Counter
        # Sample labels to get distribution (don't load all at once)
        sample_size = min(10000, self.length)
        indices = np.random.choice(self.length, sample_size, replace=False)
        labels = []
        for idx in indices:
            label = self.y_file[self.y_key][idx]
            # Convert numpy array/scalar to int
            if isinstance(label, (np.ndarray, np.generic)):
                label = int(label.item())
            else:
                label = int(label)
            labels.append(label)
        label_counts = Counter(labels)
        return {self.idx_to_class.get(int(idx), f'class_{idx}'): count for idx, count in label_counts.items()}
    
    def __len__(self):
        """Return dataset size."""
        return self.length
    
    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (image, label) where image is PIL Image and label is torch.Tensor
        """
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")
        
        # Load image from HDF5
        img_array = self.x_file[self.x_key][idx]
        
        # Convert numpy array to PIL Image
        # PCAM images are typically RGB, shape (H, W, 3) or (3, H, W)
        if img_array.shape[0] == 3:
            # CHW format, convert to HWC
            img_array = np.transpose(img_array, (1, 2, 0))
        
        # Ensure values are in [0, 255] range and uint8
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='RGB')
        
        # Load label from HDF5
        label = self.y_file[self.y_key][idx]
        # Convert to integer if needed
        if isinstance(label, (np.ndarray, np.generic)):
            label = int(label.item())
        else:
            label = int(label)
        
        label = torch.tensor(label, dtype=torch.long)
        
        # Ensure image is 224x224 (resize if needed)
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        return img, label
    
    def __del__(self):
        """Close HDF5 files when dataset is deleted."""
        if hasattr(self, 'x_file') and self.x_file is not None:
            try:
                self.x_file.close()
            except:
                pass
        if hasattr(self, 'y_file') and self.y_file is not None:
            try:
                self.y_file.close()
            except:
                pass

# -------------------------------
# DataLoader creation
# -------------------------------


def create_pcam_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4
):
    """
    Create train, valid, and test DataLoaders for PCAM dataset.

    Args:
        data_root: str - Root directory containing HDF5 files
        batch_size: int - Batch size for DataLoaders (default: 32)
        num_workers: int - Number of data loading workers (default: 4)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = PCAMDataset(
        data_root=data_root,
        split='train'
    )
    
    valid_dataset = PCAMDataset(
        data_root=data_root,
        split='valid'
    )
    
    test_dataset = PCAMDataset(
        data_root=data_root,
        split='test'
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PCAM dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing HDF5 files')
    args = parser.parse_args()
    
    # Test dataset directly (without DataLoader)
    train_dataset = PCAMDataset(data_root=args.data_root, split='train')
    valid_dataset = PCAMDataset(data_root=args.data_root, split='valid')
    test_dataset = PCAMDataset(data_root=args.data_root, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a single sample
    img, label = train_dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"Label: {label.item()}, Class: {train_dataset.idx_to_class[label.item()]}")
    
    # Test DataLoader creation
    train_loader, valid_loader, test_loader = create_pcam_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        num_workers=0
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Valid loader batches: {len(valid_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"First few labels: {labels[:5]}")
        break
