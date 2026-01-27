"""
CRC Survival Analysis Dataset - GitHub Release Version

This module provides dataset classes and utilities for CRC survival analysis.
The dataset discretizes survival days into n_bins using uncensored events (qcut)
and returns survival dict including Y (1-based bin index).

Dataset structure:
  <feats_path>/
    - {slide_id}.*.h5 (patch-level features for each slide)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <labels_file>
    - CSV file with columns: slide_id, survival_days, censorship, etc.

Key characteristics:
  - Survival analysis task with discrete-time hazard modeling.
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Discretizes survival times into n_bins using quantiles of uncensored events.
  - Converts survival_days to survival_months (1 month = 30.44 days).
  - Custom collate function handles variable patch counts per slide.

Usage example:
  from crc_survival_dataset import CRCSurvivalDataset, create_crc_survival_dataloaders
  
  train_loader, val_loader, test_loader = create_crc_survival_dataloaders(
      feats_path='/path/to/features',
      splits_dir='/path/to/splits',
      labels_file='/path/to/labels.csv',
      split_idx=0,
      batch_size=8,
      n_bins=4
  )
"""
import os
import glob
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# -------------------------------
# Dataset class
# -------------------------------
class CRCSurvivalDataset(Dataset):
    """
    CRC survival dataset that discretizes survival days into n_bins using uncensored events (qcut).
    
    Returns survival dict including Y (1-based bin index).
    """
    def __init__(self, feats_path: str, splits_dir: str, labels_file: str, split: str = 'train', split_idx: int = 0,
                 n_bins: int = 4, include_clinical: bool = True,
                 min_survival_days: float = 0.0, max_survival_days: float = float('inf'),
                 eps: float = 1e-6):
        """
        Initialize CRC survival dataset.

        Args:
            feats_path: str - Path to feature H5 files directory
            splits_dir: str - Path to directory containing split CSV files
            labels_file: str - Path to CSV file containing slide_id, survival_days, censorship, etc.
            split: str - 'train', 'val', or 'test'
            split_idx: int - Index of the split file to use (0-4)
            n_bins: int - Number of bins for discretization (default: 4)
            include_clinical: bool - Whether to include clinical features (default: True)
            min_survival_days: float - Minimum survival days to include (default: 0.0)
            max_survival_days: float - Maximum survival days to include (default: inf)
            eps: float - Small epsilon value for numerical stability (default: 1e-6)
        """
        self.feats_path = feats_path
        self.split = split.lower()
        self.split_idx = split_idx
        self.n_bins = int(n_bins)
        self.include_clinical = include_clinical
        self.min_survival_days = min_survival_days
        self.max_survival_days = max_survival_days
        self.eps = eps

        # load split file
        splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Split file not found: {splits_file}")
        df_splits = pd.read_csv(splits_file)

        # Handle case where test split doesn't exist
        if self.split == 'test' and 'test' not in df_splits.columns:
            raise ValueError(f"Test split requested but not available in splits file")
        col_slide = {'train': 'train', 'val': 'val', 'test': 'test'}[self.split]

        # Load labels from labels_file
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        df_labels = pd.read_csv(labels_file)

        # filter by slides in this split
        # Note: splits contain case_id, but labels contain full slide_id
        # We need to match by case_id (first part of slide_id)
        case_ids = df_splits[col_slide].dropna().tolist()
        
        # Extract case_id from slide_id in labels (first part before first dash after TCGA-XX-XXXX)
        df_labels['case_id'] = df_labels['slide_id'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')
        
        df_filtered = df_labels[df_labels['case_id'].isin(case_ids)].copy()

        # filter by survival range
        df_filtered = df_filtered[
            (df_filtered['survival_days'] >= min_survival_days) &
            (df_filtered['survival_days'] <= max_survival_days)
        ].reset_index(drop=True)

        if len(df_filtered) == 0:
            raise ValueError(f"No valid samples found for split {self.split}")

        # Convert survival_days to survival_months (approximate: 1 month = 30.44 days)
        df_filtered['survival_months'] = df_filtered['survival_days'] / 30.44
        
        # Check for NaN values and remove them
        print(f"Before NaN removal: {len(df_filtered)} samples")
        df_filtered = df_filtered.dropna(subset=['survival_days', 'censorship', 'survival_months'])
        print(f"After NaN removal: {len(df_filtered)} samples")
        
        if len(df_filtered) == 0:
            raise ValueError(f"No valid samples found for split {self.split} after removing NaN values")

        # Create survival dict for discretization
        survival_dict = {
            'survival_times': df_filtered['survival_months'].values,
            'censorship': df_filtered['censorship'].values,
            'slide_id': df_filtered['slide_id'].values
        }

        # Discretize survival times using uncensored events only
        self.Y, self.bin_edges = self._discretize_survival_times(survival_dict)

        # Store metadata
        self.slide_ids = df_filtered['slide_id'].values
        self.survival_times = df_filtered['survival_months'].values
        self.censorship = df_filtered['censorship'].values

        print(f"CRC {self.split} dataset: {len(self.slide_ids)} samples, "
              f"n_bins={self.n_bins}, survival range: {self.survival_times.min():.1f}-{self.survival_times.max():.1f} months")

    def _discretize_survival_times(self, survival_dict):
        """
        Discretize survival times into n_bins using quantiles of uncensored events.

        Args:
            survival_dict: dict - Dictionary with 'survival_times' and 'censorship' keys

        Returns:
            tuple: (Y, bin_edges) where Y is 1-based bin indices and bin_edges is array of bin edges
        """
        survival_times = survival_dict['survival_times']
        censorship = survival_dict['censorship']
        
        # Use only uncensored events for binning
        uncensored_times = survival_times[censorship == 0]
        
        if len(uncensored_times) < self.n_bins:
            print(f"Warning: Only {len(uncensored_times)} uncensored events, using all events for binning")
            times_for_binning = survival_times
        else:
            times_for_binning = uncensored_times

        # Create quantile-based bins
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = np.quantile(times_for_binning, quantiles)
        bin_edges[0] = 0  # Ensure first bin starts at 0
        bin_edges[-1] = np.inf  # Ensure last bin goes to infinity

        # Assign bin indices (1-based)
        Y = np.digitize(survival_times, bin_edges[1:], right=False) + 1
        Y = np.clip(Y, 1, self.n_bins)  # Ensure all values are in [1, n_bins]

        return Y, bin_edges

    def __len__(self):
        """Return dataset size."""
        return len(self.slide_ids)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features, survival_dict) where features is torch.Tensor and survival_dict is dict
        """
        slide_id = self.slide_ids[idx]
        
        # Load features from HDF5 file
        feat_file = os.path.join(self.feats_path, f"{slide_id}.h5")
        if not os.path.exists(feat_file):
            raise FileNotFoundError(f"Feature file not found: {feat_file}")
        
        with h5py.File(feat_file, 'r') as f:
            features = f['features'][:]  # Shape: (N_patches, feature_dim)
        
        # Convert to torch tensors
        features = torch.from_numpy(features).float()
        
        # Create survival dict
        survival_dict = {
            'Y': self.Y[idx],  # 1-based bin index
            'censorship': self.censorship[idx],  # 0=event, 1=censored
            'survival_times': self.survival_times[idx]  # survival time in months
        }
        
        return features, survival_dict

    def collate_fn(self, batch):
        """
        Custom collate function for batching.

        Args:
            batch: list of (features, survival_dict) tuples

        Returns:
            tuple: (batch_features, batch_survival_dict)
        """
        features_list, survival_dicts = zip(*batch)
        
        # Pad features to same length
        max_patches = max(f.shape[0] for f in features_list)
        padded_features = []
        
        for features in features_list:
            if features.shape[0] < max_patches:
                # Pad with zeros
                padding = torch.zeros(max_patches - features.shape[0], features.shape[1])
                padded_feat = torch.cat([features, padding], dim=0)
            else:
                padded_feat = features
            padded_features.append(padded_feat)
        
        # Stack features: (batch_size, max_patches, feature_dim)
        batch_features = torch.stack(padded_features, dim=0)
        
        # Stack survival info
        batch_survival = {
            'Y': torch.tensor([d['Y'] for d in survival_dicts], dtype=torch.long),
            'censorship': torch.tensor([d['censorship'] for d in survival_dicts], dtype=torch.float),
            'survival_times': torch.tensor([d['survival_times'] for d in survival_dicts], dtype=torch.float)
        }
        
        return batch_features, batch_survival


# -------------------------------
# DataLoader creation
# -------------------------------
def create_crc_survival_dataloaders(feats_path: str, splits_dir: str, labels_file: str, split_idx: int, 
                                  batch_size: int = 16, n_bins: int = 4, 
                                  min_survival_days: float = 0.0, max_survival_days: float = float('inf')):
    """
    Create train/val/test dataloaders for CRC survival dataset.

    Args:
        feats_path: str - Path to features directory
        splits_dir: str - Path to splits directory
        labels_file: str - Path to CSV file containing slide_id, survival_days, censorship, etc.
        split_idx: int - Split index (0-4 for 5-fold CV)
        batch_size: int - Batch size (default: 16)
        n_bins: int - Number of time bins for discretization (default: 4)
        min_survival_days: float - Minimum survival days to include (default: 0.0)
        max_survival_days: float - Maximum survival days to include (default: inf)

    Returns:
        tuple: (train_loader, val_loader, test_loader) where test_loader may be None if no test split
    """
    
    # Create datasets
    train_dataset = CRCSurvivalDataset(
        feats_path=feats_path, splits_dir=splits_dir, labels_file=labels_file, split='train', split_idx=split_idx,
        n_bins=n_bins, min_survival_days=min_survival_days, max_survival_days=max_survival_days
    )
    
    val_dataset = CRCSurvivalDataset(
        feats_path=feats_path, splits_dir=splits_dir, labels_file=labels_file, split='val', split_idx=split_idx,
        n_bins=n_bins, min_survival_days=min_survival_days, max_survival_days=max_survival_days
    )
    
    # Check if test split exists
    splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
    df_splits = pd.read_csv(splits_file)
    has_test = 'test' in df_splits.columns and not df_splits['test'].isna().all()
    
    if has_test:
        test_dataset = CRCSurvivalDataset(
            feats_path=feats_path, splits_dir=splits_dir, labels_file=labels_file, split='test', split_idx=split_idx,
            n_bins=n_bins, min_survival_days=min_survival_days, max_survival_days=max_survival_days
        )
    else:
        test_dataset = None
        print(f"No test split found for split {split_idx}, using val as test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=val_dataset.collate_fn
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn
        )
    else:
        test_loader = val_loader  # Use val as test if no test split
    
    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CRC survival dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to directory containing split CSV files')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to CSV file containing slide_id, survival_days, censorship, etc.')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoaders')
    parser.add_argument('--n_bins', type=int, default=4,
                        help='Number of bins for discretization')
    args = parser.parse_args()
    
    print("=== Testing CRC Survival Dataset ===")
    print(f"Features path: {args.feats_path}")
    print(f"Splits dir: {args.splits_dir}")
    print(f"Labels file: {args.labels_file}")
    print(f"Split index: {args.split_idx}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of bins: {args.n_bins}")
    
    # Check if paths exist
    if not os.path.exists(args.feats_path):
        print(f"ERROR: Features path does not exist: {args.feats_path}")
        exit(1)
    
    if not os.path.exists(args.splits_dir):
        print(f"ERROR: Splits directory does not exist: {args.splits_dir}")
        exit(1)
    
    if not os.path.exists(args.labels_file):
        print(f"ERROR: Labels file does not exist: {args.labels_file}")
        exit(1)
    
    try:
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader, test_loader = create_crc_survival_dataloaders(
            feats_path=args.feats_path,
            splits_dir=args.splits_dir,
            labels_file=args.labels_file,
            split_idx=args.split_idx,
            batch_size=args.batch_size,
            n_bins=args.n_bins
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Test loading a batch from each split
        print("\n=== Testing Train Loader ===")
        for i, (features, survival_dict) in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"  Features shape: {features.shape}")
            print(f"  Y shape: {survival_dict['Y'].shape}")
            print(f"  Y values: {survival_dict['Y']}")
            print(f"  Censorship shape: {survival_dict['censorship'].shape}")
            print(f"  Censorship values: {survival_dict['censorship']}")
            print(f"  Survival times shape: {survival_dict['survival_times'].shape}")
            print(f"  Survival times values: {survival_dict['survival_times']}")
            if i >= 2:  # Only test first 3 batches
                break
        
        print("\n=== Testing Val Loader ===")
        for i, (features, survival_dict) in enumerate(val_loader):
            print(f"Batch {i}:")
            print(f"  Features shape: {features.shape}")
            print(f"  Y shape: {survival_dict['Y'].shape}")
            print(f"  Y values: {survival_dict['Y']}")
            print(f"  Censorship shape: {survival_dict['censorship'].shape}")
            print(f"  Censorship values: {survival_dict['censorship']}")
            print(f"  Survival times shape: {survival_dict['survival_times'].shape}")
            print(f"  Survival times values: {survival_dict['survival_times']}")
            if i >= 1:  # Only test first 2 batches
                break
        
        print("\n=== Testing Test Loader ===")
        for i, (features, survival_dict) in enumerate(test_loader):
            print(f"Batch {i}:")
            print(f"  Features shape: {features.shape}")
            print(f"  Y shape: {survival_dict['Y'].shape}")
            print(f"  Y values: {survival_dict['Y']}")
            print(f"  Censorship shape: {survival_dict['censorship'].shape}")
            print(f"  Censorship values: {survival_dict['censorship']}")
            print(f"  Survival times shape: {survival_dict['survival_times'].shape}")
            print(f"  Survival times values: {survival_dict['survival_times']}")
            if i >= 1:  # Only test first 2 batches
                break
        
        print("\n=== Dataset Statistics ===")
        # Calculate some basic statistics
        all_y = []
        all_censorship = []
        all_survival_times = []
        
        for features, survival_dict in train_loader:
            all_y.extend(survival_dict['Y'].tolist())
            all_censorship.extend(survival_dict['censorship'].tolist())
            all_survival_times.extend(survival_dict['survival_times'].tolist())
        
        all_y = np.array(all_y)
        all_censorship = np.array(all_censorship)
        all_survival_times = np.array(all_survival_times)
        
        print(f"Train set statistics:")
        print(f"  Total samples: {len(all_y)}")
        print(f"  Y range: {all_y.min()} - {all_y.max()}")
        print(f"  Y distribution: {np.bincount(all_y)}")
        print(f"  Censorship: {np.sum(all_censorship)} censored, {np.sum(1-all_censorship)} events")
        print(f"  Survival times range: {all_survival_times.min():.1f} - {all_survival_times.max():.1f} months")
        print(f"  Mean survival time: {all_survival_times.mean():.1f} months")
        
        print("\n✅ CRC Survival Dataset test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
