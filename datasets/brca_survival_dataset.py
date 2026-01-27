"""
BRCA Survival Analysis Dataset - GitHub Release Version

This module provides dataset classes and utilities for BRCA survival analysis.
The dataset discretizes survival months into n_bins using uncensored events (qcut)
and returns survival dict including Y (1-based bin index).

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <labels_file>
    - CSV file with columns: slide_id, case_id, survival_months, censorship, etc.

Key characteristics:
  - Survival analysis task with discrete-time hazard modeling.
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Discretizes survival times into n_bins using quantiles of uncensored events.
  - Custom collate function handles variable patch counts per slide.

Usage example:
  from brca_survival_dataset import BRCASurvivalDataset, create_brca_survival_dataloaders
  
  train_loader, val_loader, test_loader = create_brca_survival_dataloaders(
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
class BRCASurvivalDataset(Dataset):
    """
    BRCA survival dataset that discretizes survival months into n_bins using uncensored events (qcut).
    
    Returns survival dict including Y (1-based bin index).
    """
    def __init__(self, feats_path: str, splits_dir: str, labels_file: str, split: str = 'train', split_idx: int = 0,
                 n_bins: int = 4, include_clinical: bool = True,
                 min_survival_months: float = 0.0, max_survival_months: float = float('inf'),
                 eps: float = 1e-6):
        """
        Initialize BRCA survival dataset.

        Args:
            feats_path: str - Path to feature H5 files directory
            splits_dir: str - Path to directory containing split CSV files
            labels_file: str - Path to CSV file containing slide_id, case_id, survival_months, censorship, etc.
            split: str - 'train', 'val', or 'test'
            split_idx: int - Index of the split file to use (0-4)
            n_bins: int - Number of bins for discretization (default: 4)
            include_clinical: bool - Whether to include clinical features (default: True)
            min_survival_months: float - Minimum survival months to include (default: 0.0)
            max_survival_months: float - Maximum survival months to include (default: inf)
            eps: float - Small epsilon value for numerical stability (default: 1e-6)
        """
        self.feats_path = feats_path
        self.split = split.lower()
        self.split_idx = split_idx
        self.n_bins = int(n_bins)
        self.include_clinical = include_clinical
        self.min_survival_months = min_survival_months
        self.max_survival_months = max_survival_months
        self.eps = eps

        # load split file
        splits_file = os.path.join(splits_dir, f"splits_{split_idx}.csv")
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Split file not found: {splits_file}")
        df_splits = pd.read_csv(splits_file)

        col_slide = {'train': 'train', 'val': 'val', 'test': 'test'}[self.split]

        # Load labels from labels_file
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        df_labels = pd.read_csv(labels_file)

        # filter by slides in this split
        slide_ids = df_splits[col_slide].dropna().tolist()
        df_filtered = df_labels[df_labels['slide_id'].isin(slide_ids)].copy()

        # filter by survival range
        df_filtered = df_filtered[
            (df_filtered['survival_months'] >= min_survival_months) &
            (df_filtered['survival_months'] <= max_survival_months)
        ].reset_index(drop=True)

        # first, filter out samples that don't have feature files (so bins are computed from available data)
        available_idxs = []
        for idx, row in df_filtered.iterrows():
            slide_id = str(row['slide_id']).strip()
            pat1 = os.path.join(self.feats_path, f"{slide_id}.h5")
            pat2 = os.path.join(self.feats_path, "**", f"{slide_id}.h5")
            files = glob.glob(pat1)
            if not files:
                files = glob.glob(pat2, recursive=True)
            if files:
                available_idxs.append(idx)
        if len(available_idxs) < len(df_filtered):
            # warn, then reduce
            print(f"Warning: {len(df_filtered) - len(available_idxs)} slides missing feature files; excluding them before binning.")
            df_filtered = df_filtered.iloc[available_idxs].reset_index(drop=True)

        # prepare dataframe for dataset (keep full slide-level info but we'll compute bins from patient-level uncensored)
        self.df = pd.DataFrame({
            'slide_id': df_filtered['slide_id'],
            'case_id': df_filtered['case_id'],
            'survival_months': df_filtered['survival_months'],
            'censorship': df_filtered['censorship'],
            'age': df_filtered.get('age', pd.Series([0]*len(df_filtered))),
            'site': df_filtered.get('site', pd.Series(['unknown']*len(df_filtered))),
            'is_female': df_filtered.get('is_female', pd.Series([1]*len(df_filtered))),
            'histological_label': df_filtered.get('label', pd.Series(['UNK']*len(df_filtered)))
        }).reset_index(drop=True)

        if self.include_clinical:
            mutation_cols = [col for col in df_filtered.columns if col.endswith('_mut')]
            for col in mutation_cols:
                self.df[col] = df_filtered[col].values

        # Convert categorical features
        self._encode_categorical_features()

        # --- compute bin edges using uncensored patients (Generic's method) ---
        # build patient-level table (unique case_id)
        patients_df = df_filtered.drop_duplicates(subset=['case_id']).copy().reset_index(drop=True)
        # uncensored patients: censorship < 1 (same convention as Generic)
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        if len(uncensored_df) >= 1:
            # use pd.qcut on uncensored survival months to get equal-frequency bins based on observed events
            try:
                # qcut might give identical bins if insufficient unique values; handle by fallback
                disc_labels, q_bins = pd.qcut(uncensored_df['survival_months'], q=self.n_bins, retbins=True, labels=False, duplicates='drop')
            except Exception:
                q_bins = np.quantile(uncensored_df['survival_months'].values.astype(float),
                                     np.linspace(0.0, 1.0, self.n_bins + 1))
            # ensure endpoints cover full observed range (use patient-level min/max)
            q_bins = np.asarray(q_bins, dtype=float)
            q_bins[0] = patients_df['survival_months'].min() - self.eps
            q_bins[-1] = patients_df['survival_months'].max() + self.eps
            # if result has fewer than n_bins+1 edges (qcut with duplicates='drop'), pad or fallback
            if q_bins.size < (self.n_bins + 1):
                # fallback to quantile on uncensored (not ideal but safe)
                q_bins = np.quantile(patients_df['survival_months'].values.astype(float),
                                     np.linspace(0.0, 1.0, self.n_bins + 1))
                q_bins[0] = patients_df['survival_months'].min() - self.eps
                q_bins[-1] = patients_df['survival_months'].max() + self.eps
        else:
            # No uncensored events available: fallback to quantile over all patients
            print("Warning: no uncensored events found; falling back to overall quantiles for bin edges.")
            q_bins = np.quantile(patients_df['survival_months'].values.astype(float),
                                 np.linspace(0.0, 1.0, self.n_bins + 1))
            q_bins[0] = patients_df['survival_months'].min() - self.eps
            q_bins[-1] = patients_df['survival_months'].max() + self.eps

        # Ensure strictly increasing edges
        for i in range(1, len(q_bins)):
            if q_bins[i] <= q_bins[i-1]:
                q_bins[i] = q_bins[i-1] + 1e-6

        self.bin_edges = q_bins  # length n_bins+1

    def _encode_categorical_features(self):
        site_dummies = pd.get_dummies(self.df['site'], prefix='site')
        self.df = pd.concat([self.df, site_dummies], axis=1)
        hist_mapping = {'IDC': 0, 'ILC': 1, 'MDLC': 2}
        self.df['histological_numeric'] = self.df['histological_label'].map(hist_mapping).fillna(0).astype(int)
        self.df = self.df.fillna(0)

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def _month_to_Ybin(self, months: float) -> int:
        """
        Convert survival months to bin index (1-based).

        Args:
            months: float - Survival months

        Returns:
            int: Bin index (1..n_bins)
        """
        # Use bin_edges computed from uncensored patients
        # bin indices 1..n_bins
        # np.digitize behavior: we use internal edges (exclude min&max)
        internal_edges = self.bin_edges[1:-1].tolist()  # n_bins-1 internal thresholds
        # right=True to match Generic: intervals like (edge_{i-1}, edge_i]
        bin_idx = int(np.digitize(months, internal_edges, right=True)) + 1
        bin_idx = max(1, min(self.n_bins, bin_idx))
        return bin_idx

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features, survival_dict) where features is torch.Tensor and survival_dict is dict
        """
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id']).strip()
        pat1 = os.path.join(self.feats_path, f"{slide_id}.h5")
        pat2 = os.path.join(self.feats_path, "**", f"{slide_id}.h5")
        files = glob.glob(pat1)
        if not files:
            files = glob.glob(pat2, recursive=True)
        if not files:
            return None, None
        h5_file = files[0]
        with h5py.File(h5_file, 'r') as f:
            feats = torch.from_numpy(f['features'][:])

        months = float(row['survival_months'])
        Y_bin = self._month_to_Ybin(months)  # 1..n_bins (1-based)
        surv = {
            'survival_months': torch.tensor(months, dtype=torch.float32),
            'censorship': torch.tensor(row['censorship'], dtype=torch.float32),
            'Y': torch.tensor(Y_bin, dtype=torch.long),
            'age': torch.tensor(row['age'], dtype=torch.float32),
            'is_female': torch.tensor(row['is_female'], dtype=torch.float32),
            'histological_numeric': torch.tensor(row['histological_numeric'], dtype=torch.long),
            'case_id': row['case_id']
        }
        if self.include_clinical:
            site_cols = [col for col in self.df.columns if col.startswith('site_')]
            surv['site_features'] = torch.tensor([row[col] for col in site_cols], dtype=torch.float32)
            mutation_cols = [col for col in self.df.columns if col.endswith('_mut')]
            if mutation_cols:
                surv['mutation_features'] = torch.tensor([row[col] for col in mutation_cols], dtype=torch.float32)

        return feats, surv

# -------------------------------
# DataLoader creation
# -------------------------------
def create_brca_survival_dataloaders(feats_path: str, splits_dir: str, labels_file: str, split_idx: int = 0,
                                    batch_size: int = 8, random_state: int = 42,
                                    n_bins: int = 4, include_clinical: bool = True,
                                    min_survival_months: float = 0.0, max_survival_months: float = float('inf')):
    """
    Create train, val, test DataLoaders for BRCA survival dataset.

    Args:
        feats_path: str - Path to feature H5 files directory
        splits_dir: str - Path to directory containing split CSV files
        labels_file: str - Path to CSV file containing slide_id, case_id, survival_months, censorship, etc.
        split_idx: int - Index of the split file to use (0-4) (default: 0)
        batch_size: int - Batch size for DataLoaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)
        n_bins: int - Number of bins for discretization (default: 4)
        include_clinical: bool - Whether to include clinical features (default: True)
        min_survival_months: float - Minimum survival months to include (default: 0.0)
        max_survival_months: float - Maximum survival months to include (default: inf)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    def survival_collate_fn(batch):
        """
        Custom collate function for survival dataset.
        
        Args:
            batch: list of (features, survival_dict) tuples
        
        Returns:
            tuple: (features_list, batch_survival_dict)
        """
        valid = [(f, s) for f, s in batch if f is not None and s is not None]
        if not valid:
            return None, None
        feats_list, survs = zip(*valid)
        survival_times = torch.stack([sd['survival_months'] for sd in survs])
        censorship = torch.stack([sd['censorship'] for sd in survs])
        Y_bins = torch.stack([sd['Y'] for sd in survs]).view(-1, 1).long()  # (B,1)
        batch_surv = {
            'survival_times': survival_times,
            'censorship': censorship,
            'Y': Y_bins,
            'ages': torch.stack([sd['age'] for sd in survs]),
            'is_female': torch.stack([sd['is_female'] for sd in survs]),
            'histological_numeric': torch.stack([sd['histological_numeric'] for sd in survs]),
            'case_ids': [sd['case_id'] for sd in survs]
        }
        if include_clinical and 'site_features' in survs[0]:
            batch_surv['site_features'] = torch.stack([sd['site_features'] for sd in survs])
        if include_clinical and 'mutation_features' in survs[0]:
            batch_surv['mutation_features'] = torch.stack([sd['mutation_features'] for sd in survs])
        return feats_list, batch_surv

    train_ds = BRCASurvivalDataset(feats_path, splits_dir, labels_file, split='train', split_idx=split_idx,
                                   n_bins=n_bins, include_clinical=include_clinical,
                                   min_survival_months=min_survival_months, max_survival_months=max_survival_months)
    val_ds = BRCASurvivalDataset(feats_path, splits_dir, labels_file, split='val', split_idx=split_idx,
                                 n_bins=n_bins, include_clinical=include_clinical,
                                 min_survival_months=min_survival_months, max_survival_months=max_survival_months)
    test_ds = BRCASurvivalDataset(feats_path, splits_dir, labels_file, split='test', split_idx=split_idx,
                                  n_bins=n_bins, include_clinical=include_clinical,
                                  min_survival_months=min_survival_months, max_survival_months=max_survival_months)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=survival_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=survival_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=survival_collate_fn, worker_init_fn=lambda _: np.random.seed(random_state))
    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BRCA survival dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to directory containing split CSV files')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to CSV file containing slide_id, case_id, survival_months, censorship, etc.')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    parser.add_argument('--n_bins', type=int, default=4,
                        help='Number of bins for discretization')
    parser.add_argument('--include_clinical', action='store_true',
                        help='Include clinical features')
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader = create_brca_survival_dataloaders(
        feats_path=args.feats_path,
        splits_dir=args.splits_dir,
        labels_file=args.labels_file,
        split_idx=args.split_idx,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        include_clinical=args.include_clinical
    )
    
    print(f"# train batches: {len(train_loader)}")
    print(f"# val batches: {len(val_loader)}")
    print(f"# test batches: {len(test_loader)}")
    
    for batch in train_loader:
        if batch is None:
            continue
        feats, surv = batch
        print("Batch ok. sample Y:", surv['Y'][:5].squeeze().tolist())
        break