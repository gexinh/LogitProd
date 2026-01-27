"""
ESCA Tissue Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the ESCA (Esophageal Carcinoma)
tissue classification task. The dataset contains 11 tissue classes for multi-class
classification.

Dataset structure:
  <data_root>/
    - VALSET4_CHA_FULL/ (train split)
        - ADVENT/
        - LAM_PROP/
        - MUSC_MUC/
        - MUSC_PROP/
        - REGR_TU/
        - SH_MAG/
        - SH_OES/
        - SUB_GL/
        - SUBMUC/
        - TUMOR/
        - ULCUS/
    - VALSET1_UKK/ (test split, merged)
    - VALSET2_WNS/ (test split, merged)
    - VALSET3_TCGA/ (test split, merged)

Key characteristics:
  - Multi-class classification task (11 classes).
  - Train split: VALSET4_CHA_FULL
  - Test split: VALSET1_UKK, VALSET2_WNS, VALSET3_TCGA merged together.
  - Images are automatically resized to 224x224 if needed.

Usage example:
  from ESCA_dataset import ESCADataset, create_esca_dataloaders
  
  dataset = ESCADataset(data_root='/path/to/ESCA', split='train')
  train_loader, test_loader = create_esca_dataloaders(
      data_root='/path/to/ESCA',
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
class ESCADataset(Dataset):
    """
    ESCA tissue classification dataset with pre-defined train/test splits.
    
    Returns image and label pairs for 11-class classification:
    ADVENT, LAM_PROP, MUSC_MUC, MUSC_PROP, REGR_TU, SH_MAG, SH_OES, SUB_GL,
    SUBMUC, TUMOR, ULCUS
    """
    
    def __init__(self, data_root, split='train'):
        """
        Initialize ESCA dataset.

        Args:
            data_root: str - Root directory containing VALSET folders
            split: str - 'train' or 'test'
        """
        self.data_root = data_root
        self.split = split.lower()
        
        # Get classes from VALSET4_CHA_FULL directory
        train_dir = os.path.join(data_root, 'VALSET4_CHA_FULL')
        if os.path.exists(train_dir):
            # Get all subdirectories as classes
            potential_classes = [d for d in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, d))]
            # Sort to ensure consistent ordering
            self.classes = sorted(potential_classes)
        else:
            # Default classes if train directory doesn't exist
            self.classes = ['ADVENT', 'LAM_PROP', 'MUSC_MUC', 'MUSC_PROP', 'REGR_TU',
                          'SH_MAG', 'SH_OES', 'SUB_GL', 'SUBMUC', 'TUMOR', 'ULCUS']
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Determine data directories based on split
        if self.split == 'train':
            # Train: only VALSET4_CHA_FULL
            data_dirs = [os.path.join(data_root, 'VALSET4_CHA_FULL')]
        elif self.split == 'test':
            # Test: merge VALSET1_UKK, VALSET2_WNS, VALSET3_TCGA
            data_dirs = [
                os.path.join(data_root, 'VALSET1_UKK'),
                os.path.join(data_root, 'VALSET2_WNS'),
                os.path.join(data_root, 'VALSET3_TCGA')
            ]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'")
        
        # Collect all image paths with labels from all data directories
        self.samples = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Data directory not found: {data_dir}")
                continue
            
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    # Some classes might not exist in all VALSET folders
                    continue
                
                # Find all images in this class
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff',
                                  '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
                for ext in image_extensions:
                    image_paths = glob.glob(os.path.join(class_dir, ext))
                    for img_path in image_paths:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for {self.split} split in {data_dirs}")
        
        print(f"ESCA {self.split.upper()} set: {len(self.samples)} samples")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """
        Get class distribution in current split.

        Returns:
            dict: Dictionary mapping class names to sample counts
        """
        from collections import Counter
        labels = [label for _, label in self.samples]
        label_counts = Counter(labels)
        # Convert to class names for readability
        return {self.idx_to_class[idx]: count for idx, count in label_counts.items()}
    
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
            # Ensure image is 224x224 (resize if needed)
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
            return img, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data
            return Image.new('RGB', (224, 224), color='black'), label

# -------------------------------
# DataLoader creation
# -------------------------------


def create_esca_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4
):
    """
    Create train and test DataLoaders for ESCA dataset.

    Args:
        data_root: str - Root directory containing VALSET folders
        batch_size: int - Batch size for DataLoaders (default: 32)
        num_workers: int - Number of data loading workers (default: 4)

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = ESCADataset(
        data_root=data_root,
        split='train'
    )
    
    test_dataset = ESCADataset(
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
    
    parser = argparse.ArgumentParser(description='Test ESCA dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing VALSET folders')
    args = parser.parse_args()
    
    # Test dataset directly (without DataLoader)
    train_dataset = ESCADataset(data_root=args.data_root, split='train')
    test_dataset = ESCADataset(data_root=args.data_root, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a single sample
    img, label = train_dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"Label: {label.item()}, Class: {train_dataset.idx_to_class[label.item()]}")
    
    # Test DataLoader creation
    train_loader, test_loader = create_esca_dataloaders(
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
