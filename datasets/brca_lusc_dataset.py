"""
BRCA+LUSC Gene Mutation Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for binary gene mutation prediction
using merged BRCA and LUSC datasets. The dataset uses pre-extracted patch-level features
stored in HDF5 format and custom CSV splits for cross-validation.

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide, unified path for both BRCA and LUSC)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <mutation_labels_file>
    - CSV file with columns: case_id, TP53, PIK3CA, PTEN, KRAS, ARID1A, NF1, BRCA2

Key characteristics:
  - Binary classification task for gene mutation prediction.
  - Supports multiple target genes: TP53, PIK3CA, PTEN, KRAS, ARID1A, NF1, BRCA2.
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Unified feature path for both BRCA and LUSC datasets.

Usage example:
  from brca_lusc_dataset import BRCALUSCClassificationDataset, create_brca_lusc_classification_dataloaders
  
  train_loader, val_loader, test_loader = create_brca_lusc_classification_dataloaders(
      feats_path='/path/to/features',
      mutation_labels_file='/path/to/mutation_labels.csv',
      splits_dir='/path/to/splits',
      target_gene='TP53',
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
from typing import List, Tuple

# -------------------------------
# Dataset class
# -------------------------------
class BRCALUSCClassificationDataset(Dataset):
    """
    Merged BRCA and LUSC classification dataset for gene mutation prediction.
    
    Binary classification for a single gene (TP53, PIK3CA, PTEN, KRAS, ARID1A, NF1, or BRCA2).
    """
    def __init__(self, feats_path: str, 
                 mutation_labels_file: str,
                 splits_dir: str, split: str = 'train', split_idx: int = 0,
                 target_gene: str = 'TP53'):
        """
        Initialize BRCA+LUSC classification dataset.

        Args:
            feats_path: str - Unified path to patch-level features directory (contains both BRCA and LUSC features)
            mutation_labels_file: str - Path to mutation labels CSV file
            splits_dir: str - Directory containing split CSV files
            split: str - 'train', 'val', or 'test'
            split_idx: int - Index of the split file to use (0-4)
            target_gene: str - Target gene for binary classification (default: 'TP53')
        """
        self.feats_path = feats_path
        self.split = split.lower()
        self.split_idx = split_idx
        self.target_gene = target_gene.upper()  # TP53, PIK3CA, PTEN, KRAS, ARID1A, NF1, or BRCA2

        # Load mutation labels file
        if not os.path.exists(mutation_labels_file):
            raise FileNotFoundError(f"Mutation labels file not found: {mutation_labels_file}")
        df_mutation_labels = pd.read_csv(mutation_labels_file)
        
        # Validate target gene
        available_genes = ['TP53', 'PIK3CA', 'PTEN', 'KRAS', 'ARID1A', 'NF1', 'BRCA2']
        if self.target_gene not in available_genes:
            raise ValueError(f"target_gene must be one of {available_genes}, got {target_gene}")
        if self.target_gene not in df_mutation_labels.columns:
            raise ValueError(f"Missing mutation column: {self.target_gene} in labels file")
        
        # Load split file (merged format: contains case_id in train/val/test columns)
        splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Split file not found: {splits_file}")
        df_splits = pd.read_csv(splits_file)

        col_split = {'train': 'train', 'val': 'val', 'test': 'test'}[self.split]
        case_ids = df_splits[col_split].dropna().tolist()

        # Match case_ids to mutation labels and find corresponding slide_ids
        matched_samples = []
        skipped_no_label = 0
        skipped_no_feat = 0
        
        for case_id in case_ids:
            case_id_str = str(case_id).strip()
            
            # Find all samples with this case_id in mutation labels
            matches = df_mutation_labels[df_mutation_labels['case_id'] == case_id_str]
            
            if len(matches) == 0:
                skipped_no_label += 1
                continue
            
            # Use the first match (since label file now only has case_id, no Tumor_Sample_Barcode)
            match_row = matches.iloc[0]
            
            # Try to find the corresponding feature file using case_id
            found_feat = False
            dataset_type = None
            slide_id_for_feat = None
            
            # Search for feature files by case_id pattern in unified path
            if case_id_str.startswith('TCGA-'):
                # Search in unified path: look for files matching case_id pattern
                pattern1 = os.path.join(self.feats_path, f"{case_id_str}*.h5")
                pattern2 = os.path.join(self.feats_path, "**", f"{case_id_str}*.h5")
                feat_files = sorted(glob.glob(pattern1))
                if not feat_files:
                    feat_files = sorted(glob.glob(pattern2, recursive=True))
                        
                if feat_files:
                    # Found feature file - use the first one
                    # Determine dataset type based on case_id or file path if needed
                    # For now, we'll use a generic type since both BRCA and LUSC are in the same path
                    dataset_type = 'BRCA_LUSC'  # Unified dataset type
                    slide_id_for_feat = os.path.basename(feat_files[0]).replace('.h5', '')
                    found_feat = True
                
            if found_feat:
                matched_samples.append({
                    'slide_id': slide_id_for_feat,  # Use the actual slide_id found in feature file
                    'case_id': case_id_str,
                    'label': int(match_row[self.target_gene])  # Binary label for target gene
                })
            else:
                skipped_no_feat += 1

        if skipped_no_label > 0:
            print(f"Warning: {skipped_no_label} case_ids not found in labels file")
        if skipped_no_feat > 0:
            print(f"Warning: {skipped_no_feat} samples skipped due to missing feature files")
        
        if len(matched_samples) == 0:
            raise ValueError(f"No samples matched for split {self.split} (split_idx={split_idx}). "
                           f"Skipped: {skipped_no_label} no label, {skipped_no_feat} no feature file")
        
        # Create dataframe from matched samples
        # Note: matched_samples already filtered by feature file existence
        self.df = pd.DataFrame(matched_samples)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples with valid feature files found for split {self.split} (split_idx={split_idx})")

        pos_count = (self.df['label'] == 1).sum()
        neg_count = (self.df['label'] == 0).sum()
        print(f"BRCA+LUSC Binary Classification ({self.target_gene}) {self.split} set (split_{split_idx}): {len(self.df)} samples")
        print(f"Label distribution: {pos_count} positive (mutated), {neg_count} negative (wild-type) ({100*pos_count/len(self.df):.1f}% positive)")

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
        
        # Load features from unified path
        # Search for {slide_id}.h5 in the unified features directory
        pat1 = os.path.join(self.feats_path, f"{slide_id}.h5")
        pat2 = os.path.join(self.feats_path, "**", f"{slide_id}.h5")
        files = sorted(glob.glob(pat1))
        if not files:
            files = sorted(glob.glob(pat2, recursive=True))
        
        if not files:
            return None, None
        
        h5_file = files[0]
        with h5py.File(h5_file, 'r') as f:
            # Try to find the correct key for features
            features = None
            # First, try common keys
            for key in ['features', 'patches', 'feats']:
                if key in f.keys():
                    data = f[key][:]
                    # Check if this looks like feature data (2D array with reasonable dimensions)
                    if len(data.shape) == 2 and data.shape[1] >= 100:
                        features = data
                        break
                    elif len(data.shape) == 2 and data.shape[0] >= 100:
                        # Might be transposed, check if transposing makes sense
                        if data.shape[0] > data.shape[1] and data.shape[1] < 10:
                            # Likely transposed: (feature_dim, n_patches) -> (n_patches, feature_dim)
                            features = data.T
                            break
            
            # If no suitable key found, try the first key but check dimensions
            if features is None:
                keys = list(f.keys())
                for key in keys:
                    data = f[key][:]
                    if len(data.shape) == 2:
                        # Check if dimensions make sense
                        if data.shape[1] >= 100:
                            features = data
                            break
                        elif data.shape[0] >= 100 and data.shape[1] < 10:
                            # Likely transposed
                            features = data.T
                            break
            
            if features is None:
                print(f"Warning: Could not find valid features for {slide_id}, available keys: {list(f.keys())}")
                return None, None
            
            feats = torch.from_numpy(features).float()
            # Ensure features are 2D: (n_patches, feature_dim)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            elif feats.dim() > 2:
                feats = feats.view(-1, feats.shape[-1])
            
            # Check if feature dimension is reasonable (should be >= 100)
            if feats.shape[1] < 100:
                print(f"Warning: Feature dimension too small for {slide_id}: {feats.shape}, skipping")
                return None, None

        # Create binary classification label for target gene
        label = torch.tensor(int(row['label']), dtype=torch.long)  # 0 or 1
        
        label_dict = {
            'label': label,  # Binary label: 0 (wild-type) or 1 (mutated)
            'case_id': row['case_id'],
            'slide_id': slide_id
        }

        return feats, label_dict

# -------------------------------
# DataLoader creation
# -------------------------------
def create_brca_lusc_classification_dataloaders(feats_path: str,
                                                 mutation_labels_file: str,
                                                 splits_dir: str, split_idx: int = 0,
                                                 batch_size: int = 8, random_state: int = 42,
                                                 target_gene: str = 'TP53'):
    """
    Create dataloaders for binary classification of a single gene mutation.

    Args:
        feats_path: str - Unified path to patch-level features directory (contains both BRCA and LUSC features)
        mutation_labels_file: str - Path to mutation labels CSV file
        splits_dir: str - Directory containing split CSV files
        split_idx: int - Split index for cross-validation (default: 0)
        batch_size: int - Batch size for dataloaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)
        target_gene: str - Which gene to predict ('TP53', 'PIK3CA', 'PTEN', 'KRAS', 'ARID1A', 'NF1', or 'BRCA2') (default: 'TP53')

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    def classification_collate_fn(batch):
        """
        Custom collate function for gene mutation classification.
        
        Args:
            batch: list of (features, label_dict) tuples
        
        Returns:
            tuple: (features_list, batch_labels_dict)
        """
        valid = [(f, l) for f, l in batch if f is not None and l is not None]
        if not valid:
            return None, None
        feats_list, label_dicts = zip(*valid)
        # Stack labels: (B,) for binary classification
        labels = torch.stack([ld['label'] for ld in label_dicts])  # (B,)
        batch_labels = {
            'labels': labels,  # (B,) binary classification
            'case_ids': [ld['case_id'] for ld in label_dicts],
            'slide_ids': [ld['slide_id'] for ld in label_dicts]
        }
        return feats_list, batch_labels

    train_ds = BRCALUSCClassificationDataset(feats_path, mutation_labels_file,
                                             splits_dir, split='train', split_idx=split_idx,
                                             target_gene=target_gene)
    val_ds = BRCALUSCClassificationDataset(feats_path, mutation_labels_file,
                                           splits_dir, split='val', split_idx=split_idx,
                                           target_gene=target_gene)
    test_ds = BRCALUSCClassificationDataset(feats_path, mutation_labels_file,
                                            splits_dir, split='test', split_idx=split_idx,
                                            target_gene=target_gene)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=classification_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=classification_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=classification_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BRCA+LUSC dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Unified path to patch-level features directory')
    parser.add_argument('--mutation_labels_file', type=str, required=True,
                        help='Path to mutation labels CSV file')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing split CSV files')
    parser.add_argument('--target_gene', type=str, default='TP53',
                        choices=['TP53', 'PIK3CA', 'PTEN', 'KRAS', 'ARID1A', 'NF1', 'BRCA2'],
                        help='Target gene for binary classification')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    args = parser.parse_args()
    
    print(f"Testing split_{args.split_idx} for {args.target_gene} binary classification...")
    print(f"Using mutation labels file: {args.mutation_labels_file}")
    print(f"Using splits directory: {args.splits_dir}")
    
    train_loader, val_loader, test_loader = create_brca_lusc_classification_dataloaders(
        feats_path=args.feats_path,
        mutation_labels_file=args.mutation_labels_file,
        splits_dir=args.splits_dir,
        split_idx=args.split_idx,
        batch_size=args.batch_size,
        target_gene=args.target_gene
    )
    
    print(f"# train batches: {len(train_loader)}")
    print(f"# val batches: {len(val_loader)}")
    print(f"# test batches: {len(test_loader)}")
    
    for batch in train_loader:
        if batch is None:
            continue
        feats, labels = batch
        print(f"Split {args.split_idx} - Batch ok. Labels shape:", labels['labels'].shape)
        print(f"Split {args.split_idx} - Sample labels ({args.target_gene}):", labels['labels'][:10].tolist())
        break

