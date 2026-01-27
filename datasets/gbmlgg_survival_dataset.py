"""
GBMLGG Survival Analysis Dataset - GitHub Release Version

This module provides dataset classes and utilities for merged TCGA-GBM / TCGA-LGG
survival analysis. The dataset discretizes survival months into n_bins using uncensored
events and returns survival dict including Y (1-based bin index).

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide)
  <splits_dir>/
    - splits_0.csv, splits_1.csv, ..., splits_4.csv (cross-validation splits)
  <merged_csv>
    - CSV file with columns: Study, case_id, survival_months, censorship, slide_id, Fold 0~4

Key characteristics:
  - Survival analysis task with discrete-time hazard modeling.
  - Uses pre-extracted patch-level features from HDF5 files.
  - Supports 5-fold cross-validation via split CSV files.
  - Discretizes survival times into n_bins using quantiles of uncensored events.
  - slide_id can be semicolon-separated multiple paths; uses first existing h5 in feats_path.
  - Custom collate function handles variable patch counts per slide.

Usage example:
  from gbmlgg_survival_dataset import GBMLGGSurvivalDataset, create_gbmlgg_survival_dataloaders
  
  train_loader, val_loader, test_loader = create_gbmlgg_survival_dataloaders(
      feats_path='/path/to/features',
      splits_dir='/path/to/splits',
      merged_csv='/path/to/merged.csv',
      split_idx=0,
      batch_size=16,
      n_bins=4
  )
"""
import os
import csv
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Dataset class
# -------------------------------
class GBMLGGSurvivalDataset(Dataset):
    """
    Survival dataset for merged TCGA-GBM / TCGA-LGG cohorts using the merged split CSV.
    
    CSV requirements (see TCGA_GBMLGG_Splits_merged.csv):
        Study, case_id, survival_months, censorship, slide_id, Fold 0~4
    - slide_id can be semicolon-separated multiple paths; uses first existing h5 in feats_path.
    - Feature naming convention in feats_path: <slide_id>.h5
    """

    def __init__(
        self,
        feats_path: str,
        merged_csv: str,
        split_csv_dir: str,
        split_idx: int,
        split: str,
        n_bins: int = 4,
        feature_key: str = "features",
        feature_ext: str = ".h5",
    ):
        """
        Initialize GBMLGG survival dataset.

        Args:
            feats_path: str - Feature h5 directory (e.g., features_conch_v15)
            merged_csv: str - Merged labels and paths CSV (TCGA_GBMLGG_Splits_merged.csv)
            split_csv_dir: str - Survival split directory (contains splits_*.csv)
            split_idx: int - Split index 0-4
            split: str - 'train', 'val', or 'test'
            n_bins: int - Number of survival bins (default: 4)
            feature_key: str - H5 internal dataset key (default: 'features')
            feature_ext: str - Feature file extension (default: '.h5')
        """
        assert split in {"train", "val", "test"}
        self.feats_path = feats_path
        self.split = split
        self.n_bins = n_bins
        self.feature_key = feature_key
        self.feature_ext = feature_ext

        # Load merged labels table
        data = self._load_csv(merged_csv)
        labels_by_case = {r["case_id"]: r for r in data}

        # Load survival split file
        split_file = Path(split_csv_dir) / f"splits_{split_idx}.csv"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        split_rows = self._load_split(split_file)
        case_ids = [cid for cid in split_rows.get(split, []) if cid]

        # Filter samples for current split and check feature existence
        cleaned = []
        for cid in case_ids:
            r = labels_by_case.get(cid)
            if r is None:
                continue
            slide_candidates = self._expand_slide_ids(r["slide_id"])
            slide_hit = None
            for sid in slide_candidates:
                fpath = os.path.join(self.feats_path, sid + self.feature_ext)
                if os.path.exists(fpath):
                    slide_hit = sid
                    break
            if slide_hit is None:
                continue
            cleaned.append(
                {
                    "case_id": cid,
                    "survival_months": float(r["survival_months"]),
                    "censorship": float(r["censorship"]),
                    "slide_id": slide_hit,
                }
            )

        if not cleaned:
            raise ValueError(f"No samples for split {split_idx} / {split}")

        self.case_ids = np.array([r["case_id"] for r in cleaned])
        self.slide_ids = np.array([r["slide_id"] for r in cleaned])
        self.survival_months = np.array([r["survival_months"] for r in cleaned], dtype=float)
        self.censorship = np.array([r["censorship"] for r in cleaned], dtype=float)

        # Bin discretization (use quantiles of uncensored events, fallback to all if insufficient)
        uncensored = self.survival_months[self.censorship == 0]
        times_for_bin = uncensored if len(uncensored) >= n_bins else self.survival_months
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(times_for_bin, quantiles)
        bin_edges[0] = 0
        bin_edges[-1] = np.inf
        self.bin_edges = bin_edges
        bins = np.digitize(self.survival_months, bin_edges[1:], right=False) + 1
        self.Y = np.clip(bins, 1, n_bins)

    @staticmethod
    def _load_csv(path: str) -> List[dict]:
        rows = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)
        return rows

    @staticmethod
    def _load_split(path: Path) -> dict:
        """
        Load split file and return dict: {'train': [...], 'val': [...], 'test': [...]}

        Args:
            path: Path - Path to split CSV file

        Returns:
            dict: Dictionary with 'train', 'val', 'test' keys containing case_id lists
        """
        res = {"train": [], "val": [], "test": []}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                for k in res.keys():
                    if k in r:
                        res[k].append(r[k])
        return res

    @staticmethod
    def _expand_slide_ids(s: str) -> List[str]:
        """
        Expand semicolon-separated slide IDs and return basenames without extensions.

        Args:
            s: str - Semicolon-separated slide ID string

        Returns:
            list: List of slide ID basenames
        """
        parts = [p.strip() for p in s.split(";") if p.strip()]
        # Get basename without extension
        return [Path(p).stem for p in parts]

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
        feat_file = os.path.join(self.feats_path, slide_id + self.feature_ext)
        if not os.path.exists(feat_file):
            raise FileNotFoundError(f"Feature file not found: {feat_file}")
        with h5py.File(feat_file, "r") as f:
            if self.feature_key in f:
                feats = f[self.feature_key][:]
            else:
                # fallback: first dataset
                key = list(f.keys())[0]
                feats = f[key][:]
        feats = torch.from_numpy(feats).float()
        # Compatible with train_abmil_save_slide_features.py: provide both survival_months and survival_times
        survival = {
            "Y": self.Y[idx],
            "censorship": self.censorship[idx],
            "survival_months": self.survival_months[idx],
            "survival_times": self.survival_months[idx],
        }
        return feats, survival

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching.

        Args:
            batch: list of (features, survival_dict) tuples

        Returns:
            tuple: (batch_features, batch_survival_dict)
        """
        feats_list, surv_list = zip(*batch)
        max_patches = max(f.shape[0] for f in feats_list)
        feat_dim = feats_list[0].shape[1]
        padded = []
        for f in feats_list:
            if f.shape[0] < max_patches:
                pad = torch.zeros(max_patches - f.shape[0], feat_dim)
                f = torch.cat([f, pad], dim=0)
            padded.append(f)
        batch_feats = torch.stack(padded, dim=0)
        batch_surv = {
            "Y": torch.tensor([s["Y"] for s in surv_list], dtype=torch.long),
            "censorship": torch.tensor([s["censorship"] for s in surv_list], dtype=torch.float),
            "survival_months": torch.tensor([s["survival_months"] for s in surv_list], dtype=torch.float),
            # Also provide survival_times for downstream compatibility
            "survival_times": torch.tensor([s["survival_months"] for s in surv_list], dtype=torch.float),
        }
        return batch_feats, batch_surv


def create_gbmlgg_survival_dataloaders(
    feats_path: str,
    merged_csv: str,
    split_csv_dir: str,
    split_idx: int,
    batch_size: int = 16,
    n_bins: int = 4,
    num_workers: int = 4,
    feature_key: str = "features",
    feature_ext: str = ".h5",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = GBMLGGSurvivalDataset(
        feats_path, merged_csv, split_csv_dir, split_idx, "train", n_bins=n_bins, feature_key=feature_key, feature_ext=feature_ext
    )
    val_ds = GBMLGGSurvivalDataset(
        feats_path, merged_csv, split_csv_dir, split_idx, "val", n_bins=n_bins, feature_key=feature_key, feature_ext=feature_ext
    )
    test_ds = GBMLGGSurvivalDataset(
        feats_path, merged_csv, split_csv_dir, split_idx, "test", n_bins=n_bins, feature_key=feature_key, feature_ext=feature_ext
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, collate_fn=train_ds.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_ds.collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=test_ds.collate_fn
    )
    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GBMLGG survival dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--merged_csv', type=str, required=True,
                        help='Path to merged CSV file (TCGA_GBMLGG_Splits_merged.csv)')
    parser.add_argument('--split_csv_dir', type=str, required=True,
                        help='Directory containing split CSV files')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoaders')
    parser.add_argument('--n_bins', type=int, default=4,
                        help='Number of bins for discretization')
    args = parser.parse_args()
    
    loaders = create_gbmlgg_survival_dataloaders(
        feats_path=args.feats_path,
        merged_csv=args.merged_csv,
        split_csv_dir=args.split_csv_dir,
        split_idx=args.split_idx,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        num_workers=0,
    )
    for name, loader in zip(["train", "val", "test"], loaders):
        print(f"{name}: {len(loader)} batches")
        for feats, surv in loader:
            print(f"features shape: {feats.shape}, Y shape: {surv['Y'].shape}")
            break

