"""
CRC-100K Tissue Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the CRC-100K tissue classification
task. The dataset contains 9 tissue classes for multi-class classification with
pre-defined train/test splits.

Dataset structure:
  <data_root>/
    - NCT-CRC-HE-100K/  (train split)
        - ADI/
        - BACK/
        - DEB/
        - LYM/
        - MUC/
        - MUS/
        - NORM/
        - STR/
        - TUM/
    - CRC-VAL-HE-7K/  (test split)
        - ADI/
        - BACK/
        - DEB/
        - LYM/
        - MUC/
        - MUS/
        - NORM/
        - STR/
        - TUM/

Key characteristics:
  - Multi-class classification task (9 classes).
  - Pre-defined train/test splits.
  - Images are loaded as PIL Images.

Usage example:
  from crc_100k_dataset import CRCDataset, create_crc_dataloaders
  
  dataset = CRCDataset(data_root='/path/to/CRC-100K', split='train')
  train_loader, test_loader = create_crc_dataloaders(
      data_root='/path/to/CRC-100K',
      batch_size=32,
      num_workers=4
  )
"""
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Dataset class
# -------------------------------
class CRCDataset(Dataset):
    """
    CRC-100K tissue classification dataset with pre-defined train/test splits.
    
    Returns image and label pairs for 9-class classification:
    ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
    """
    
    def __init__(self, data_root, split='train'):
        """
        Initialize CRC-100K dataset.

        Args:
            data_root: str - Root directory containing NCT-CRC-HE-100K and CRC-VAL-HE-7K folders
            split: str - 'train' or 'test'
        """
        self.data_root = data_root
        self.split = split.lower()
        
        # Determine data directory based on split
        if self.split == 'train':
            self.data_dir = os.path.join(data_root, 'NCT-CRC-HE-100K')
        elif self.split == 'test':
            self.data_dir = os.path.join(data_root, 'CRC-VAL-HE-7K')
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'")
        
        # Class names (9 classes)
        self.classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all image paths with labels
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Find all images in this class
            image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', 
                              '*.TIF', '*.TIFF', '*.PNG', '*.JPG', '*.JPEG']
            for ext in image_extensions:
                image_paths = glob.glob(os.path.join(class_dir, ext))
                for img_path in image_paths:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        print(f"CRC-100K {self.split} set: {len(self.samples)} samples")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """
        Get class distribution in current split.

        Returns:
            dict: Dictionary mapping class indices to sample counts
        """
        from collections import Counter
        labels = [label for _, label in self.samples]
        return dict(Counter(labels))
    
    def __len__(self):
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (image, label) where image is PIL Image and label is torch.Tensor
        """
        image_path, label = self.samples[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        # Load image
        try:
            img = Image.open(image_path).convert('RGB')
            return img, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data
            return Image.new('RGB', (224, 224), color='black'), label

# -------------------------------
# DataLoader creation
# -------------------------------
def create_crc_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4
):
    """
    Create train and test DataLoaders for CRC-100K dataset.

    Args:
        data_root: str - Root directory containing NCT-CRC-HE-100K and CRC-VAL-HE-7K folders
        batch_size: int - Batch size for DataLoaders (default: 32)
        num_workers: int - Number of data loading workers (default: 4)

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = CRCDataset(
        data_root=data_root,
        split='train'
    )
    
    test_dataset = CRCDataset(
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CRC-100K dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing NCT-CRC-HE-100K and CRC-VAL-HE-7K folders')
    args = parser.parse_args()
    
    # Test dataset directly (without DataLoader)
    train_dataset = CRCDataset(data_root=args.data_root, split='train')
    test_dataset = CRCDataset(data_root=args.data_root, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a single sample
    img, label = train_dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"Label: {label.item()}, Class: {train_dataset.idx_to_class[label.item()]}")
    
    # Test DataLoader creation
    train_loader, test_loader = create_crc_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        num_workers=0
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"First few labels: {labels[:5]}")
        break

