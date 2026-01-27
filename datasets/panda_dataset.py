"""
PANDA WSI Classification Dataset - GitHub Release Version

This module provides dataset classes and utilities for the PANDA (Prostate cANcer
grading Assessment) WSI classification task. The dataset uses pre-extracted patch-level
features stored in HDF5 format and a single split file for train/val/test splits.

Dataset structure:
  <feats_path>/
    - {slide_id}.h5 (patch-level features for each slide)
  <splits_dir>/
    - PANDA/splits_0_k.csv (single split file with columns: train, train_label, val, val_label, test, test_label)

Key characteristics:
  - Multi-class classification task (Gleason score grading).
  - Uses pre-extracted patch-level features from HDF5 files.
  - Single split file format (splits_0_k.csv) with labels included.
  - Custom collate function handles variable patch counts per slide.

Usage example:
  from panda_dataset import PANDADataset, create_panda_dataloaders
  
  train_loader, val_loader, test_loader = create_panda_dataloaders(
      feats_path='/path/to/features',
      splits_dir='/path/to/splits',
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
class PANDADataset(Dataset):
    """
    PANDA dataset using precomputed slide-level features (one .h5 per slide).
    
    Uses a single split file: PANDA/splits_0_k.csv with columns:
    train, train_label, val, val_label, test, test_label
    """

    def __init__(self, feats_path, splits_dir, split: str = "train"):
        """
        Initialize PANDA dataset.

        Args:
            feats_path: str - Path to feature H5 files directory
            splits_dir: str - Directory containing PANDA/splits_0_k.csv
            split: str - 'train', 'val', or 'test'
        """
        self.feats_path = feats_path
        self.split = split.lower()

        # Read the single split file: splits_0_k.csv
        split_file = os.path.join(splits_dir, "PANDA", "splits_0_k.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        df_split = pd.read_csv(split_file)

        col_slide = {
            "train": "train",
            "val": "val",
            "test": "test",
        }[self.split]
        col_label = f"{col_slide}_label"
        if col_slide not in df_split.columns or col_label not in df_split.columns:
            raise ValueError(
                f"Columns '{col_slide}' and '{col_label}' must exist in {split_file}"
            )

        # Get slide_ids and labels for current split (remove NaN)
        slide_ids = df_split[col_slide].dropna().astype(str).tolist()
        labels = df_split[col_label].dropna().tolist()
        if len(slide_ids) != len(labels):
            # Theoretically should be paired in the same row, mismatch indicates CSV issue
            min_len = min(len(slide_ids), len(labels))
            slide_ids = slide_ids[:min_len]
            labels = labels[:min_len]

        self.df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "label": labels,
            }
        ).reset_index(drop=True)

        # Filter out slides that don't have feature files
        available_indices = []
        for idx, row in self.df.iterrows():
            slide_id = row["slide_id"]
            feat_file = os.path.join(self.feats_path, f"{slide_id}.h5")
            if os.path.exists(feat_file):
                available_indices.append(idx)

        if len(available_indices) < len(self.df):
            print(
                f"Warning: {len(self.df) - len(available_indices)} PANDA slides don't have feature files in {self.feats_path}, filtering them out"
            )
            self.df = self.df.iloc[available_indices].reset_index(drop=True)

        # Print dataset information
        print(f"PANDA {self.split} set: {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features, label) where features is torch.Tensor and label is torch.Tensor
        """
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"]).strip()

        # PANDA feature naming convention: <slide_id>.h5
        feat_file = os.path.join(self.feats_path, f"{slide_id}.h5")
        if not os.path.exists(feat_file):
            print(f"Warning: Feature file not found for slide {slide_id}")
            return None, None

        with h5py.File(feat_file, "r") as f:
            features = torch.from_numpy(f["features"][:])

        label = torch.tensor(row["label"], dtype=torch.float32)
        return features, label

# -------------------------------
# DataLoader creation
# -------------------------------
def create_panda_dataloaders(
    feats_path,
    splits_dir,
    batch_size: int = 8,
    random_state: int = 42,
):
    """
    Create train, val, test DataLoaders for PANDA dataset (single-model).

    Args:
        feats_path: str - Path to feature H5 files directory
        splits_dir: str - Directory containing PANDA/splits_0_k.csv
        batch_size: int - Batch size for DataLoaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    def collate_fn(batch):
        """
        Custom collate function to filter out None values and handle variable patch counts.
        
        Args:
            batch: list of (features, label) tuples
        
        Returns:
            tuple: (features_list, labels_tensor)
        """
        # Filter None
        valid_batch = [
            (features, label)
            for features, label in batch
            if features is not None and label is not None
        ]
        if not valid_batch:
            return None, None

        features_list, labels_list = zip(*valid_batch)
        return features_list, torch.stack(labels_list)

    train_ds = PANDADataset(feats_path, splits_dir, split="train")
    val_ds = PANDADataset(feats_path, splits_dir, split="val")
    test_ds = PANDADataset(feats_path, splits_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )

    return train_loader, val_loader, test_loader


# -------------------------------
# Multi-model dataset class
# -------------------------------
class MultiModelPANDADataset(Dataset):
    """
    Multi-model PANDA dataset for embedding fusion.
    
    Loads features from multiple models and supports different fusion strategies.
    Interface consistent with MultiModelBRACSDataset / MultiModelBRCADataset.
    """

    def __init__(
        self,
        model_feat_dirs,
        splits_dir,
        split: str = "train",
        merge_strategy: str = "concat",
        cluster_k: int | None = None,
        rank: int = 0,
    ):
        """
        Initialize multi-model PANDA dataset.

        Args:
            model_feat_dirs: list of str - Embedding directory for each model (features_MODEL)
            splits_dir: str - PANDA split directory (contains PANDA/splits_0_k.csv)
            split: str - 'train', 'val', or 'test'
            merge_strategy: str - 'concat', 'sum', or 'cluster' (default: 'concat')
            cluster_k: int or None - Number of clusters (used only for cluster strategy) (default: None)
            rank: int - Process rank, only rank==0 prints info (default: 0)
        """
        self.model_feat_dirs = model_feat_dirs
        self.split = split.lower()
        self.merge_strategy = merge_strategy
        self.cluster_k = cluster_k
        self.rank = rank

        # Still use only splits_0_k.csv
        split_file = os.path.join(splits_dir, "PANDA", "splits_0_k.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        df_split = pd.read_csv(split_file)

        col_slide = {
            "train": "train",
            "val": "val",
            "test": "test",
        }[self.split]
        col_label = f"{col_slide}_label"
        if col_slide not in df_split.columns or col_label not in df_split.columns:
            raise ValueError(
                f"Columns '{col_slide}' and '{col_label}' must exist in {split_file}"
            )

        slide_ids = df_split[col_slide].dropna().astype(str).tolist()
        labels = df_split[col_label].dropna().tolist()
        if len(slide_ids) != len(labels):
            min_len = min(len(slide_ids), len(labels))
            slide_ids = slide_ids[:min_len]
            labels = labels[:min_len]

        self.df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "label": labels,
            }
        ).reset_index(drop=True)

        # Cluster strategy: concatenate all slides' multi-model features and perform KMeans once
        if self.merge_strategy == "cluster":
            all_embeds = []
            for idx in range(len(self.df)):
                slide_id = self.df.iloc[idx]["slide_id"]
                embeds = []
                for feat_dir in self.model_feat_dirs:
                    feat_file = os.path.join(feat_dir, f"{slide_id}.h5")
                    if not os.path.exists(feat_file):
                        continue
                    with h5py.File(feat_file, "r") as f:
                        embeds.append(f["features"][:])
                if len(embeds) == len(self.model_feat_dirs):
                    concat_emb = np.concatenate(embeds, axis=1)
                    all_embeds.append(concat_emb)
            if all_embeds:
                all_embeds = np.vstack(all_embeds)
                self.kmeans = KMeans(n_clusters=cluster_k).fit(all_embeds)
            else:
                self.kmeans = None
        else:
            self.kmeans = None

        if self.rank == 0:
            print(f"MultiModelPANDA {self.split} set: {len(self.df)} samples")
            print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")

    def __len__(self):
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (features_list, label) where features_list is list of torch.Tensor and label is torch.Tensor
        """
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        embeds = []
        for feat_dir in self.model_feat_dirs:
            feat_file = os.path.join(feat_dir, f"{slide_id}.h5")
            if not os.path.exists(feat_file):
                print(
                    f"Warning: Feature file not found for slide {slide_id} in {feat_dir}"
                )
                return None, None
            with h5py.File(feat_file, "r") as f:
                embeds.append(f["features"][:])

        if len(embeds) != len(self.model_feat_dirs):
            return None, None

        # Return separated features for each model, let the upper model decide how to fuse
        return [torch.from_numpy(emb).float() for emb in embeds], label

# -------------------------------
# Multi-model collate function
# -------------------------------
def multimodel_panda_collate_fn(batch):
    """
    Custom collate function for multi-model PANDA dataset.
    
    Args:
        batch: list of (features_list, label) tuples
    
    Returns:
        tuple: (features_list, labels_tensor)
    """
    valid_batch = [
        (features, label)
        for features, label in batch
        if features is not None and label is not None
    ]
    if not valid_batch:
        return None, None
    features_list, labels_list = zip(*valid_batch)
    return features_list, torch.stack(labels_list)

def create_multimodel_panda_dataloaders(
    model_feat_dirs,
    splits_dir,
    batch_size: int = 8,
    random_state: int = 42,
    merge_strategy: str = "concat",
    cluster_k: int | None = None,
):
    """
    Create multi-model train, val, test DataLoaders for PANDA dataset.
    
    Interface consistent with create_multimodel_brca_dataloaders.

    Args:
        model_feat_dirs: list of str - Embedding directory for each model
        splits_dir: str - PANDA split directory (contains PANDA/splits_0_k.csv)
        batch_size: int - Batch size for DataLoaders (default: 8)
        random_state: int - Random seed for reproducibility (default: 42)
        merge_strategy: str - 'concat', 'sum', or 'cluster' (default: 'concat')
        cluster_k: int or None - Number of clusters (used only for cluster strategy) (default: None)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    train_ds = MultiModelPANDADataset(
        model_feat_dirs,
        splits_dir,
        split="train",
        merge_strategy=merge_strategy,
        cluster_k=cluster_k,
    )
    val_ds = MultiModelPANDADataset(
        model_feat_dirs,
        splits_dir,
        split="val",
        merge_strategy=merge_strategy,
        cluster_k=cluster_k,
    )
    test_ds = MultiModelPANDADataset(
        model_feat_dirs,
        splits_dir,
        split="test",
        merge_strategy=merge_strategy,
        cluster_k=cluster_k,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodel_panda_collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodel_panda_collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodel_panda_collate_fn,
        worker_init_fn=lambda _: np.random.seed(random_state),
    )

    return train_loader, val_loader, test_loader


# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PANDA dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing PANDA/splits_0_k.csv')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = create_panda_dataloaders(
        feats_path=args.feats_path,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size
    )
    print(f"# train batches: {len(train_loader)}")
    print(f"# val batches:   {len(val_loader)}")
    print(f"# test batches:  {len(test_loader)}")


