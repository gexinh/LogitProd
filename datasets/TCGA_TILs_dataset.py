"""
TCGA-TILs Tissue Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the TCGA-TILs (Tumor Infiltrating
Lymphocytes) tissue classification task. The dataset contains binary classification
(TIL-positive vs TIL-negative) with pre-defined train/val/test splits.

Dataset structure:
  <data_root>/
    - images-tcga-tils/
        - pancancer/
            - train/
                - til-negative/
                - til-positive/
            - val/
                - til-negative/
                - til-positive/
            - test/
                - til-negative/
                - til-positive/

Key characteristics:
  - Binary classification task (TIL-positive vs TIL-negative).
  - Pre-defined train/val/test splits.
  - Fixed to pancancer (all cancer types combined).
  - Images are loaded as PIL Images.

Usage example:
  from TCGA_TILs_dataset import TCGATILsDataset, create_tcga_tils_dataloaders
  
  dataset = TCGATILsDataset(data_root='/path/to/TCGA-TILs', split='train')
  train_loader, val_loader, test_loader = create_tcga_tils_dataloaders(
      data_root='/path/to/TCGA-TILs',
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
class TCGATILsDataset(Dataset):
    """
    TCGA-TILs tissue classification dataset with pre-defined train/val/test splits.
    
    Returns image and label pairs for binary classification: TIL-positive (1) vs TIL-negative (0)
    """
    
    def __init__(self, data_root, split='train'):
        """
        Initialize TCGA-TILs dataset.

        Args:
            data_root: str - Root directory containing images-tcga-tils folder
            split: str - 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.split = split.lower()
        
        # Class names (binary classification)
        self.classes = ['til-negative', 'til-positive']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Determine data directory (fixed to pancancer)
        images_dir = os.path.join(data_root, 'images-tcga-tils')
        self.data_dir = os.path.join(images_dir, 'pancancer', self.split)
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
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
        
        print(f"TCGA-TILs (pancancer) {self.split.upper()} set: {len(self.samples)} samples")
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
def create_tcga_tils_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4
):
    """
    Create train, val, and test DataLoaders for TCGA-TILs dataset.

    Args:
        data_root: str - Root directory containing images-tcga-tils folder
        batch_size: int - Batch size for DataLoaders (default: 32)
        num_workers: int - Number of data loading workers (default: 4)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = TCGATILsDataset(
        data_root=data_root,
        split='train'
    )
    
    val_dataset = TCGATILsDataset(
        data_root=data_root,
        split='val'
    )
    
    test_dataset = TCGATILsDataset(
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
    
    val_loader = DataLoader(
        val_dataset,
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
    
    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TCGA-TILs dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing images-tcga-tils folder')
    args = parser.parse_args()
    
    # Test dataset directly (without DataLoader)
    train_dataset = TCGATILsDataset(data_root=args.data_root, split='train')
    val_dataset = TCGATILsDataset(data_root=args.data_root, split='val')
    test_dataset = TCGATILsDataset(data_root=args.data_root, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a single sample
    img, label = train_dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"Label: {label.item()}, Class: {train_dataset.idx_to_class[label.item()]}")
    
    # Test DataLoader creation
    train_loader, val_loader, test_loader = create_tcga_tils_dataloaders(
        data_root=args.data_root,
        batch_size=32,
        num_workers=0
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"First few labels: {labels[:5]}")
        break

