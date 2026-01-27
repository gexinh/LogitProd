"""
CCRCC Tissue Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the CCRCC (Clear Cell Renal
Cell Carcinoma) tissue classification task. The dataset contains 4 tissue classes
for multi-class classification.

Dataset structure:
  <data_root>/
    - tissue_classification/
        - blood/
        - cancer/
        - normal/
        - stroma/

Key characteristics:
  - Multi-class classification task (4 classes: cancer, stroma, normal, blood).
  - Train/test split: 8:2 ratio (stratified split).
  - Images are loaded as PIL Images.

Usage example:
  from CCRCC_dataset import CCRCCDataset, create_ccrcc_dataloaders
  
  dataset = CCRCCDataset(data_root='/path/to/CCRCC', split='train')
  train_loader, test_loader = create_ccrcc_dataloaders(
      data_root='/path/to/CCRCC',
      batch_size=32,
      num_workers=4
  )
"""
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------------------
# Dataset class
# -------------------------------
class CCRCCDataset(Dataset):
    """
    CCRCC tissue classification dataset with 8:2 train/test split.
    
    Returns image and label pairs for 4-class classification:
    cancer, stroma, normal, blood
    """
    
    def __init__(self, data_root, split='train', test_size=0.2, random_state=42):
        """
        Initialize CCRCC dataset.

        Args:
            data_root: str - Root directory containing tissue_classification folder
            split: str - 'train' or 'test'
            test_size: float - Proportion of data to use for test set (default: 0.2)
            random_state: int - Random seed for reproducibility (default: 42)
        """
        self.data_root = data_root
        self.split = split.lower()
        
        # Class names (only four categories: cancer, stroma, normal, blood)
        self.classes = ['cancer', 'stroma', 'normal', 'blood']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all image paths with labels
        self.samples = []
        tissue_dir = os.path.join(data_root, 'tissue_classification')
        
        for class_name in self.classes:
            class_dir = os.path.join(tissue_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Find all images in this class
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            for ext in image_extensions:
                image_paths = glob.glob(os.path.join(class_dir, ext))
                for img_path in image_paths:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {tissue_dir}")
        
        print(f"Total samples found: {len(self.samples)}")
        
        # Split into train and test
        train_samples, test_samples = train_test_split(
            self.samples,
            test_size=test_size,
            random_state=random_state,
            stratify=[label for _, label in self.samples]  # Stratified split
        )
        
        if self.split == 'train':
            self.samples = train_samples
        elif self.split == 'test':
            self.samples = test_samples
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'")
        
        print(f"CCRCC {self.split} set: {len(self.samples)} samples")
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


def create_ccrcc_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4,
    test_size=0.2,
    random_state=42
):
    """
    Create train and test DataLoaders for CCRCC dataset.

    Args:
        data_root: str - Root directory containing tissue_classification folder
        batch_size: int - Batch size for DataLoaders (default: 32)
        num_workers: int - Number of data loading workers (default: 4)
        test_size: float - Proportion of data to use for test set (default: 0.2)
        random_state: int - Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = CCRCCDataset(
        data_root=data_root,
        split='train',
        test_size=test_size,
        random_state=random_state
    )
    
    test_dataset = CCRCCDataset(
        data_root=data_root,
        split='test',
        test_size=test_size,
        random_state=random_state
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
    
    parser = argparse.ArgumentParser(description='Test CCRCC dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing tissue_classification folder')
    args = parser.parse_args()
    
    # Test dataset directly (without DataLoader)
    train_dataset = CCRCCDataset(data_root=args.data_root, split='train')
    test_dataset = CCRCCDataset(data_root=args.data_root, split='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a single sample
    img, label = train_dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"Label: {label.item()}, Class: {train_dataset.idx_to_class[label.item()]}")
    
    # Test DataLoader creation
    train_loader, test_loader = create_ccrcc_dataloaders(
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

