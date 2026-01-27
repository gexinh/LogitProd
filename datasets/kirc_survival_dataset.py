"""
KIRC Survival Analysis Dataset - GitHub Release Version

This module provides dataset classes and utilities for KIRC survival analysis.
The dataset discretizes survival days into n_bins using uncensored events (qcut)
and returns survival dict including Y (1-based bin index).

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <labels_file>
    - CSV file with columns: slide_id, case_id, survival_days, censorship, etc.

Key characteristics:
  - Survival analysis task with discrete-time hazard modeling.
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Discretizes survival times into n_bins using quantiles of uncensored events.
  - Converts survival_days to survival_months (1 month = 30.44 days).
  - Custom collate function handles variable patch counts per slide.

Usage example:
  from kirc_survival_dataset import KircSurvivalDataset, create_kirc_survival_dataloaders
  
  train_loader, val_loader, test_loader = create_kirc_survival_dataloaders(
      feats_path='/path/to/features',
      splits_dir='/path/to/splits',
      labels_file='/path/to/labels.csv',
      split_idx=0,
      batch_size=16,
      n_bins=4
  )
"""
import os
import pandas as pd
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# Dataset class
# -------------------------------
class KircSurvivalDataset(Dataset):
    """
    KIRC survival dataset that discretizes survival days into n_bins using uncensored events.
    
    Returns survival dict including Y (1-based bin index).
    """
    def __init__(self, feats_path: str, labels_csv: str, split_file: str, split: str, n_bins: int = 4):
        """
        Initialize KIRC survival dataset.

        Args:
            feats_path: str - Path to features directory
            labels_csv: str - Path to labels CSV file
            split_file: str - Path to split CSV file
            split: str - 'train', 'val', or 'test'
            n_bins: int - Number of time bins for discrete survival analysis (default: 4)
        """
        self.feats_path = feats_path
        self.n_bins = n_bins
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        # Remove samples with NaN in critical columns
        self.labels_df = self.labels_df.dropna(subset=['survival_days', 'censorship'])
        
        # Load split
        self.split_df = pd.read_csv(split_file)
        
        # Get case_ids for this split
        if split in self.split_df.columns:
            case_ids = self.split_df[split].dropna().tolist()
        else:
            raise ValueError(f"Split '{split}' not found in split file")
        
        # Filter labels by case_ids
        self.labels_df = self.labels_df[self.labels_df['case_id'].isin(case_ids)]
        
        if len(self.labels_df) == 0:
            raise ValueError(f"No valid samples found for split {split}")
        
        # Convert survival_days to survival_months (divide by 30.44)
        self.labels_df = self.labels_df.copy()
        self.labels_df['survival_months'] = self.labels_df['survival_days'] / 30.44
        
        # Create discrete time bins using quantiles of uncensored events (same as BLCA)
        survival_times = self.labels_df['survival_months'].values
        censorship = self.labels_df['censorship'].values
        
        # Use only uncensored events for binning
        uncensored_times = survival_times[censorship == 0]
        
        if len(uncensored_times) < n_bins:
            print(f"Warning: Only {len(uncensored_times)} uncensored events, using all events for binning")
            times_for_binning = survival_times
        else:
            times_for_binning = uncensored_times
        
        # Create quantile-based bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(times_for_binning, quantiles)
        bin_edges[0] = 0  # Ensure first bin starts at 0
        bin_edges[-1] = np.inf  # Ensure last bin goes to infinity
        
        # Assign bin indices (1-based, same as BLCA)
        self.labels_df['time_bin'] = np.digitize(survival_times, bin_edges[1:], right=False) + 1
        self.labels_df['time_bin'] = np.clip(self.labels_df['time_bin'], 1, n_bins)
        
        self.bin_edges = bin_edges
        
        # Before storing metadata, filter out samples without existing feature files (.h5)
        def _feat_exists(slide_id: str) -> bool:
            feat_file = os.path.join(self.feats_path, f"{slide_id}.h5")
            return os.path.exists(feat_file)

        exists_mask = self.labels_df['slide_id'].apply(_feat_exists)
        num_missing = int((~exists_mask).sum())
        if num_missing > 0:
            logger.warning(f"KIRC: skipping {num_missing} slides with missing feature files under {self.feats_path}")
        self.labels_df = self.labels_df[exists_mask].reset_index(drop=True)

        # Store metadata (same as BLCA)
        self.slide_ids = self.labels_df['slide_id'].values
        self.survival_times = self.labels_df['survival_months'].values
        self.censorship = self.labels_df['censorship'].values
        self.Y = self.labels_df['time_bin'].values
        
        logger.info(f"KIRC {split} dataset: {len(self.slide_ids)} samples")
        logger.info(f"Bin edges: {self.bin_edges}")
        logger.info(f"Censorship rate: {self.censorship.mean():.3f}")
    
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
        
        # Load features from HDF5 file (same as BLCA)
        feat_file = os.path.join(self.feats_path, f"{slide_id}.h5")
        if not os.path.exists(feat_file):
            raise FileNotFoundError(f"Feature file not found: {feat_file}")
        
        with h5py.File(feat_file, 'r') as f:
            features = f['features'][:]  # Shape: (N_patches, feature_dim)
        
        # Convert to torch tensors (same as BLCA)
        features = torch.from_numpy(features).float()
        
        # Create survival dict (same as BLCA)
        survival_dict = {
            'Y': self.Y[idx],  # 1-based bin index
            'censorship': self.censorship[idx],  # 0=event, 1=censored
            'survival_times': self.survival_times[idx]  # survival time in months
        }
        
        return features, survival_dict

    def collate_fn(self, batch):
        """
        Custom collate function for batching (align with BLCA):
        - Pad variable-length patch features to max length in batch
        - Return tuple: (batch_features, batch_survival_dict)

        Args:
            batch: list of (features, survival_dict) tuples

        Returns:
            tuple: (batch_features, batch_survival_dict)
        """
        features_list, survival_dicts = zip(*batch)

        # Pad features to the same number of patches in this batch
        max_patches = max(f.shape[0] for f in features_list)
        feature_dim = features_list[0].shape[1]

        padded_features = []
        for features in features_list:
            if features.shape[0] < max_patches:
                padding = torch.zeros(max_patches - features.shape[0], feature_dim)
                features = torch.cat([features, padding], dim=0)
            padded_features.append(features)

        # Stack features -> (batch_size, max_patches, feature_dim)
        batch_features = torch.stack(padded_features, dim=0)

        # Stack survival info into a dict
        batch_survival = {
            'Y': torch.tensor([d['Y'] for d in survival_dicts], dtype=torch.long),
            'censorship': torch.tensor([d['censorship'] for d in survival_dicts], dtype=torch.float),
            'survival_times': torch.tensor([d['survival_times'] for d in survival_dicts], dtype=torch.float)
        }

        # Keep the same output structure as BLCA
        return batch_features, batch_survival

# -------------------------------
# DataLoader creation
# -------------------------------
def create_kirc_survival_dataloaders(
    feats_path: str,
    splits_dir: str,
    labels_file: str,
    split_idx: int,
    batch_size: int = 16,
    n_bins: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create KIRC survival data loaders for train/val/test.

    Args:
        feats_path: str - Path to features directory
        splits_dir: str - Directory containing split files
        labels_file: str - Path to CSV file containing slide_id, case_id, survival_days, censorship, etc.
        split_idx: int - Split index (0-4 for 5-fold CV)
        batch_size: int - Batch size (default: 16)
        n_bins: int - Number of time bins (default: 4)
        num_workers: int - Number of worker processes (default: 4)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    split_file = os.path.join(splits_dir, f'splits_{split_idx}.csv')
    
    # Create datasets
    train_ds = KircSurvivalDataset(feats_path, labels_file, split_file, 'train', n_bins)
    val_ds = KircSurvivalDataset(feats_path, labels_file, split_file, 'val', n_bins)
    
    # For KIRC, use test split if available, otherwise use val
    if os.path.exists(split_file):
        test_ds = KircSurvivalDataset(feats_path, labels_file, split_file, 'test', n_bins)
    else:
        test_ds = val_ds
    
    # Create data loaders (explicitly pass dataset-specific collate_fn)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False,
        collate_fn=train_ds.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
        collate_fn=val_ds.collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
        collate_fn=test_ds.collate_fn
    )
    
    return train_loader, val_loader, test_loader

# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test KIRC survival dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to directory containing split CSV files')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to CSV file containing slide_id, case_id, survival_days, censorship, etc.')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoaders')
    parser.add_argument('--n_bins', type=int, default=4,
                        help='Number of bins for discretization')
    args = parser.parse_args()
    
    try:
        train_loader, val_loader, test_loader = create_kirc_survival_dataloaders(
            feats_path=args.feats_path,
            splits_dir=args.splits_dir,
            labels_file=args.labels_file,
            split_idx=args.split_idx,
            batch_size=args.batch_size,
            n_bins=args.n_bins
        )
        print('✅ KIRC dataset loading successful!')
        print(f'Train batches: {len(train_loader)}')
        print(f'Val batches: {len(val_loader)}')
        print(f'Test batches: {len(test_loader)}')
        
        # Test a batch
        for batch in train_loader:
            features, survival_dict = batch
            print(f"Batch features shape: {features.shape}")
            print(f"Batch Y shape: {survival_dict['Y'].shape}")
            print(f"Batch censorship shape: {survival_dict['censorship'].shape}")
            print(f"Batch survival_times shape: {survival_dict['survival_times'].shape}")
            break
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
