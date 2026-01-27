"""
BRCA WSI Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the BRCA (Breast Cancer)
WSI classification task. The dataset uses pre-extracted patch-level features stored
in HDF5 format and custom CSV splits for cross-validation.

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <labels_file>
    - CSV file with columns: slide_id, case_id, label

Key characteristics:
  - Binary classification task (IDC vs ILC).
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Custom collate function handles variable patch counts per slide.
  - Returns label dictionary with both histological_label and label_numeric.

Usage example:
  from brca_dataset import BRCADataset, create_brca_dataloaders
  
  train_loader, val_loader, test_loader = create_brca_dataloaders(
      feats_path='/path/to/features',
      splits_dir='/path/to/splits',
      labels_file='/path/to/labels.csv',
      split_idx=0,
      batch_size=8
  )
"""
import os
import glob
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

# -------------------------------
# Dataset class
# -------------------------------
class BRCADataset(Dataset):
    """
    BRCA dataset class using custom CSV splits.
    
    Loads patch-level features from HDF5 files and labels from CSV files.
    Returns label dictionary with both histological_label and label_numeric.
    """
    def __init__(self, feats_path, splits_dir, labels_file, split='train', split_idx=0):
        """
        Initialize BRCA dataset.

        Args:
            feats_path: str - Path to feature H5 files directory
            splits_dir: str - Path to directory containing split CSV files (splits_0.csv to splits_4.csv)
            labels_file: str - Path to CSV file containing slide_id, case_id, and label columns
            split: str - One of 'train', 'val', 'test'
            split_idx: int - Index of the split file to use (0-4)
        """
        self.feats_path = feats_path
        self.split = split.lower()
        self.split_idx = split_idx

        # Construct splits file path
        splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
        
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Split file not found: {splits_file}")

        # Load split annotations
        df_splits = pd.read_csv(splits_file)
        
        # Get slide IDs for the specified split
        col_slide = {
            'train': 'train',
            'val':   'val',
            'test':  'test'
        }[self.split]
        
        # Load labels from labels_file
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        df_labels = pd.read_csv(labels_file)
        
        # Get slide IDs for current split
        slide_ids = df_splits[col_slide].dropna().tolist()
        
        # Filter labels for slides in current split
        df_filtered = df_labels[df_labels['slide_id'].isin(slide_ids)]
        
        # Prepare DataFrame of slides and labels
        # For BRCA, we only use histological type (IDC, ILC)
        self.df = pd.DataFrame({
            'slide_id': df_filtered['slide_id'],
            'case_id': df_filtered['case_id'],
            'histological_label': df_filtered['label']
        }).reset_index(drop=True)
        
        # Convert histological labels to numeric for training
        # IDC -> 0, ILC -> 1
        label_mapping = {'IDC': 0, 'ILC': 1}
        self.df['label_numeric'] = self.df['histological_label'].map(label_mapping)
        
        # Check for any unmapped labels
        unmapped = self.df[self.df['label_numeric'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: Found unmapped labels: {unmapped['histological_label'].unique()}")
            # Remove unmapped labels
            self.df = self.df.dropna(subset=['label_numeric']).reset_index(drop=True)

        # Filter out slides that don't have feature files
        available_slides = []
        for idx, row in self.df.iterrows():
            slide_id = str(row['slide_id']).strip()
            # Check if feature file exists
            feature_file = os.path.join(self.feats_path, f"{slide_id}.h5")
            if os.path.exists(feature_file):
                available_slides.append(idx)
        
        if len(available_slides) < len(self.df):
            print(f"Warning: {len(self.df) - len(available_slides)} slides don't have feature files, filtering them out")
            self.df = self.df.iloc[available_slides].reset_index(drop=True)

        print(f"BRCA {self.split} set (split_{split_idx}): {len(self.df)} samples")
        print(f"Numeric label distribution: {self.df['label_numeric'].value_counts().to_dict()}")

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features, label_dict) where features is torch.Tensor and label_dict is dict
        """
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id']).strip()

        # Try multiple matching strategies to handle with/without UUID and subfolders
        patterns = [
            os.path.join(self.feats_path, f"{slide_id}.h5"),  # Exact match since slide_id already has UUID
            os.path.join(self.feats_path, "**", f"{slide_id}.h5"),  # Search in subdirectories
        ]

        matching_files = []
        for pat in patterns:
            files = glob.glob(pat, recursive=True)
            if files:
                matching_files = files
                break

        if not matching_files:
            print(f"Warning: Feature file not found for slide {slide_id}")
            return None, None  # Return None to indicate missing file
        
        # Use the first matching file (should be only one)
        h5_file = matching_files[0]
        
        with h5py.File(h5_file, 'r') as f:
            features = torch.from_numpy(f['features'][:])

        # Create comprehensive label dictionary
        label_dict = {
            'histological_label': row['histological_label'],
            'label_numeric': torch.tensor(row['label_numeric'], dtype=torch.float32)
        }

        return features, label_dict

    def get_label_for_training(self, idx):
        """
        Get numeric label for training (compatible with training scripts).

        Args:
            idx: int - Index of the item

        Returns:
            float: Numeric label value
        """
        row = self.df.iloc[idx]
        return row['label_numeric']

# -------------------------------
# DataLoader creation
# -------------------------------
def create_brca_dataloaders(feats_path, splits_dir, labels_file, split_idx=0,
                           batch_size=8, random_state=42):
    """
    Create train, val, test DataLoaders based on custom CSV splits.

    Args:
        feats_path: str - Path to feature H5 files directory
        splits_dir: str - Path to directory containing split CSV files
        labels_file: str - Path to CSV file containing slide_id, case_id, and label columns
        split_idx: int - Index of the split file to use (0-4) (default: 0)
        batch_size: int - Batch size for DataLoaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Custom collate function to filter out None values and handle variable patch counts
    def collate_fn(batch):
        """
        Custom collate function to filter out None values and handle variable patch counts.
        
        Args:
            batch: list of (features, label_dict) tuples
        
        Returns:
            tuple: (features_list, labels_list)
        """
        # Filter out None values
        valid_batch = [(features, label) for features, label in batch if features is not None and label is not None]
        if not valid_batch:
            return None, None
        
        # Separate features and labels
        features_list, labels_list = zip(*valid_batch)
        
        # Since the number of patches per slide may be different, we need to handle each sample separately
        return features_list, labels_list

    # Instantiate datasets
    train_ds = BRCADataset(feats_path, splits_dir, labels_file, split='train', split_idx=split_idx)
    val_ds   = BRCADataset(feats_path, splits_dir, labels_file, split='val', split_idx=split_idx)
    test_ds  = BRCADataset(feats_path, splits_dir, labels_file, split='test', split_idx=split_idx)

    # DataLoaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))

    return train_loader, val_loader, test_loader


# -------------------------------
# Multi-model dataset class
# -------------------------------
class MultiModelBRCADataset(Dataset):
    """
    Multi-model BRCA dataset for embedding fusion.
    
    Loads features from multiple models and supports different fusion strategies.
    """
    def __init__(self, model_feat_dirs, splits_dir, labels_file, split='train', split_idx=0, merge_strategy='concat', cluster_k=None, rank=0):
        """
        Initialize multi-model BRCA dataset.

        Args:
            model_feat_dirs: list of str - Embedding directory for each model
            splits_dir: str - Path to directory containing split CSV files
            labels_file: str - Path to CSV file containing slide_id, case_id, and label columns
            split: str - 'train'/'val'/'test'
            split_idx: int - Index of the split file to use (0-4)
            merge_strategy: str - 'concat'/'sum'/'cluster' (default: 'concat')
            cluster_k: int - Number of clusters (used only for cluster strategy) (default: None)
            rank: int - Process rank, only rank0 prints info (default: 0)
        """
        self.model_feat_dirs = model_feat_dirs
        self.split = split.lower()
        self.split_idx = split_idx
        self.merge_strategy = merge_strategy
        self.cluster_k = cluster_k
        self.rank = rank

        # Construct splits file path
        splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
        
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Split file not found: {splits_file}")

        # Load split annotations
        df_splits = pd.read_csv(splits_file)
        
        # Get slide IDs for the specified split
        col_slide = {
            'train': 'train',
            'val':   'val',
            'test':  'test'
        }[self.split]
        
        # Load labels from labels_file
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        df_labels = pd.read_csv(labels_file)
        
        # Get slide IDs for current split
        slide_ids = df_splits[col_slide].dropna().tolist()
        
        # Filter labels for slides in current split
        df_filtered = df_labels[df_labels['slide_id'].isin(slide_ids)]
        
        # Prepare DataFrame of slides and labels
        self.df = pd.DataFrame({
            'slide_id': df_filtered['slide_id'],
            'case_id': df_filtered['case_id'],
            'histological_label': df_filtered['label']
        }).reset_index(drop=True)
        
        # Convert histological labels to numeric for training
        # IDC -> 0, ILC -> 1
        label_mapping = {'IDC': 0, 'ILC': 1}
        self.df['label_numeric'] = self.df['histological_label'].map(label_mapping)
        
        # Check for any unmapped labels
        unmapped = self.df[self.df['label_numeric'].isna()]
        if len(unmapped) > 0:
            print(f"Warning: Found unmapped labels: {unmapped['histological_label'].unique()}")
            # Remove unmapped labels
            self.df = self.df.dropna(subset=['label_numeric']).reset_index(drop=True)

        # Clustering strategy: fit embeddings of all slides
        if self.merge_strategy == 'cluster':
            all_embeds = []
            for idx in range(len(self.df)):
                slide_id = self.df.iloc[idx]['slide_id']
                embeds = []
                for feat_dir in self.model_feat_dirs:
                    # Look for feature file with UUID suffix
                    pattern = os.path.join(feat_dir, f"{slide_id}.*.h5")
                    matching_files = glob.glob(pattern)
                    if not matching_files:
                        continue
                    h5_file = matching_files[0]
                    with h5py.File(h5_file, 'r') as f:
                        embeds.append(f['features'][:])
                if len(embeds) == len(self.model_feat_dirs):
                    concat_emb = np.concatenate(embeds, axis=1)
                    all_embeds.append(concat_emb)
            if all_embeds:
                all_embeds = np.vstack(all_embeds)
                self.kmeans = KMeans(n_clusters=cluster_k).fit(all_embeds)
            else:
                self.kmeans = None

        # Only print info in rank0
        if self.rank == 0:
            print(f"MultiModelBRCA {self.split} set (split_{split_idx}): {len(self.df)} samples")
            print(f"Numeric label distribution: {self.df['label_numeric'].value_counts().to_dict()}")

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features_list, label_dict) where features_list is list of torch.Tensor and label_dict is dict
        """
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id']).strip()
        embeds = []
        for feat_dir in self.model_feat_dirs:
            # Try multiple matching strategies to handle with/without UUID and subfolders
            patterns = [
                os.path.join(feat_dir, f"{slide_id}.h5"),  # Exact match since slide_id already has UUID
                os.path.join(feat_dir, "**", f"{slide_id}.h5"),  # Search in subdirectories
            ]

            matching_files = []
            for pat in patterns:
                files = glob.glob(pat, recursive=True)
                if files:
                    matching_files = files
                    break
            if not matching_files:
                print(f"Warning: Feature file not found for slide {slide_id} in {feat_dir}")
                return None, None
            h5_file = matching_files[0]
            with h5py.File(h5_file, 'r') as f:
                embeds.append(f['features'][:])
        if len(embeds) != len(self.model_feat_dirs):
            return None, None

        # Create comprehensive label dictionary
        label_dict = {
            'histological_label': row['histological_label'],
            'label_numeric': torch.tensor(row['label_numeric'], dtype=torch.float32)
        }

        # Return separated features, let the model handle fusion
        return [torch.from_numpy(emb).float() for emb in embeds], label_dict

# -------------------------------
# Multi-model collate function
# -------------------------------
def multimodel_brca_collate_fn(batch):
    """
    Custom collate function for multi-model BRCA dataset.
    
    Args:
        batch: list of (features_list, label_dict) tuples
    
    Returns:
        tuple: (features_list, labels_list)
    """
    valid_batch = [(features, label) for features, label in batch if features is not None and label is not None]
    if not valid_batch:
        return None, None
    features_list, labels_list = zip(*valid_batch)
    # features_list: batch * n_models * [num_patches, in_dim]
    # Since the number of patches per slide may be different, we need to handle each sample separately
    return features_list, labels_list

def create_multimodel_brca_dataloaders(model_feat_dirs, splits_dir, labels_file, split_idx=0,
                                      batch_size=8, random_state=42,
                                      merge_strategy='concat', cluster_k=None):
    """
    Create multi-model train, val, test DataLoaders for BRCA dataset.

    Args:
        model_feat_dirs: list of str - Embedding directory for each model
        splits_dir: str - Path to directory containing split CSV files
        labels_file: str - Path to CSV file containing slide_id, case_id, and label columns
        split_idx: int - Index of the split file to use (0-4) (default: 0)
        batch_size: int - Batch size for DataLoaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)
        merge_strategy: str - 'concat'/'sum'/'cluster' (default: 'concat')
        cluster_k: int - Number of clusters (used only for cluster strategy) (default: None)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Instantiate datasets
    train_ds = MultiModelBRCADataset(model_feat_dirs, splits_dir, labels_file, split='train', 
                                    split_idx=split_idx, merge_strategy=merge_strategy, cluster_k=cluster_k)
    val_ds   = MultiModelBRCADataset(model_feat_dirs, splits_dir, labels_file, split='val', 
                                    split_idx=split_idx, merge_strategy=merge_strategy, cluster_k=cluster_k)
    test_ds  = MultiModelBRCADataset(model_feat_dirs, splits_dir, labels_file, split='test', 
                                    split_idx=split_idx, merge_strategy=merge_strategy, cluster_k=cluster_k)

    # DataLoaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=multimodel_brca_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=multimodel_brca_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=multimodel_brca_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))

    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BRCA dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to directory containing split CSV files')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to CSV file containing slide_id, case_id, and label columns')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = create_brca_dataloaders(
        feats_path=args.feats_path,
        splits_dir=args.splits_dir,
        labels_file=args.labels_file,
        split_idx=args.split_idx,
        batch_size=args.batch_size
    )

    print(f"# train batches: {len(train_loader)}")
    print(f"# val batches:   {len(val_loader)}")
    print(f"# test batches:  {len(test_loader)}")

    for batch in train_loader:
        if batch is None:
            continue
        feats, labs = batch
        if feats is None or len(feats) == 0 or labs is None:
            continue
        print(f"Features: {len(feats)} slides with variable patch counts")
        # labs is a list/tuple of label dicts per slide; show keys and a sample
        print(f"Label keys: {list(labs[0].keys())}")
        print(f"Sample label - histological_label: {labs[0]['histological_label']}")
        print(f"Sample label - label_numeric: {labs[0]['label_numeric']}")
        break