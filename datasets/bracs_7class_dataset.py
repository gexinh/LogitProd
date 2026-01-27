import os
import glob
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

class BRACS7ClassDataset(Dataset):
    def __init__(self, feats_path, splits_dir, split='train', split_idx=0):
        """
        BRACS 7-class dataset class using custom CSV splits
        Args:
            feats_path: Path to feature H5 files directory
            splits_dir: Path to directory containing split CSV files (splits_0.csv to splits_4.csv)
            split: One of 'train', 'val', 'test'
            split_idx: Index of the split file to use (0-4)
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
        
        # Load labels from BRACS_7class_all.csv
        labels_file = "/home/laq/scratch/BRACS_split/BRACS_7class_all.csv"
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
            'label': df_filtered['label']
        }).reset_index(drop=True)
        
        # Create label mapping for 7 classes
        # BRACS 7 classes: ADH, DCIS, FEA, IC, N, PB, UDH
        self.label_to_idx = {
            'ADH': 0,  # Atypical Ductal Hyperplasia
            'DCIS': 1, # Ductal Carcinoma In Situ
            'FEA': 2,  # Flat Epithelial Atypia
            'IC': 3,   # Invasive Carcinoma
            'N': 4,    # Normal
            'PB': 5,   # Papillary Benign
            'UDH': 6   # Usual Ductal Hyperplasia
        }
        
        # Convert labels to numeric indices
        self.df['label_idx'] = self.df['label'].map(self.label_to_idx).astype(int)

        print(f"BRACS 7-class {self.split} set (split_{split_idx}): {len(self.df)} samples")
        print(f"7-class label distribution: {self.df['label'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        
        # Look for feature file with UUID suffix (like CRC dataset)
        pattern = os.path.join(self.feats_path, f"{slide_id}.*.h5")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Try direct filename (like Camelyon16 dataset)
            h5_file = os.path.join(self.feats_path, f"{slide_id}.h5")
            if not os.path.exists(h5_file):
                print(f"Warning: Feature file not found for slide {slide_id}")
                return None, None  # Return None to indicate missing file
        else:
            # Use the first matching file (should be only one)
            h5_file = matching_files[0]
        
        with h5py.File(h5_file, 'r') as f:
            features = torch.from_numpy(f['features'][:])

        # Use all patches, no more sampling
        # Keep all original patches

        label = torch.tensor(row['label_idx'], dtype=torch.long)  # Use long for classification
        return features, label


def create_bracs_7class_dataloaders(feats_path, splits_dir, split_idx=0,
                                   batch_size=8, random_state=42):
    """
    Create train, val, test DataLoaders for BRACS 7-class dataset based on custom CSV splits.
    Args:
        feats_path: Path to feature H5 files directory
        splits_dir: Path to directory containing split CSV files
        split_idx: Index of the split file to use (0-4)
        batch_size: Batch size for DataLoaders
        random_state: Random seed for reproducibility
    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Custom collate function to filter out None values and handle variable patch counts
    def collate_fn(batch):
        # Filter out None values
        valid_batch = [(features, label) for features, label in batch if features is not None and label is not None]
        if not valid_batch:
            return None, None
        
        # Separate features and labels
        features_list, labels_list = zip(*valid_batch)
        
        # Since the number of patches per slide may be different, we need to handle each sample separately
        return features_list, torch.stack(labels_list)

    # Instantiate datasets
    train_ds = BRACS7ClassDataset(feats_path, splits_dir, split='train', split_idx=split_idx)
    val_ds   = BRACS7ClassDataset(feats_path, splits_dir, split='val', split_idx=split_idx)
    test_ds  = BRACS7ClassDataset(feats_path, splits_dir, split='test', split_idx=split_idx)

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

# ===================== Multi-model embedding fusion version =====================
class MultiModelBRACS7ClassDataset(Dataset):
    def __init__(self, model_feat_dirs, splits_dir, split='train', split_idx=0, 
                 merge_strategy='concat', cluster_k=None, rank=0):
        """
        model_feat_dirs: list of str, embedding directory for each model
        splits_dir: Path to directory containing split CSV files
        split: 'train'/'val'/'test'
        split_idx: Index of the split file to use (0-4)
        merge_strategy: 'concat'/'sum'/'cluster'
        cluster_k: int, number of clusters (used only for cluster strategy)
        rank: int, process rank, only rank0 prints info
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
            'label': df_filtered['label']
        }).reset_index(drop=True)
        
        # Create label mapping for 7 classes
        self.label_to_idx = {
            'ADH': 0,  # Atypical Ductal Hyperplasia
            'DCIS': 1, # Ductal Carcinoma In Situ
            'FEA': 2,  # Flat Epithelial Atypia
            'IC': 3,   # Invasive Carcinoma
            'N': 4,    # Normal
            'PB': 5,   # Papillary Benign
            'UDH': 6   # Usual Ductal Hyperplasia
        }
        
        # Convert labels to numeric indices
        self.df['label_idx'] = self.df['label'].map(self.label_to_idx).astype(int)

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
                        # Try direct filename
                        h5_file = os.path.join(feat_dir, f"{slide_id}.h5")
                        if not os.path.exists(h5_file):
                            continue
                    else:
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
            print(f"MultiModelBRACS 7-class {self.split} set (split_{split_idx}): {len(self.df)} samples")
            print(f"7-class label distribution: {self.df['label'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        embeds = []
        for feat_dir in self.model_feat_dirs:
            # Look for feature file with UUID suffix
            pattern = os.path.join(feat_dir, f"{slide_id}.*.h5")
            matching_files = glob.glob(pattern)
            if not matching_files:
                # Try direct filename
                h5_file = os.path.join(feat_dir, f"{slide_id}.h5")
                if not os.path.exists(h5_file):
                    print(f"Warning: Feature file not found for slide {slide_id} in {feat_dir}")
                    return None, None
            else:
                h5_file = matching_files[0]
            with h5py.File(h5_file, 'r') as f:
                embeds.append(f['features'][:])
        if len(embeds) != len(self.model_feat_dirs):
            return None, None

        # Return separated features, let the model handle fusion
        label = torch.tensor(row['label_idx'], dtype=torch.long)  # Use long for classification
        return [torch.from_numpy(emb).float() for emb in embeds], label

# Multi-model collate_fn
def multimodel_bracs_7class_collate_fn(batch):
    valid_batch = [(features, label) for features, label in batch if features is not None and label is not None]
    if not valid_batch:
        return None, None
    features_list, labels_list = zip(*valid_batch)
    # features_list: batch * n_models * [num_patches, in_dim]
    # Since the number of patches per slide may be different, we need to handle each sample separately
    return features_list, torch.stack(labels_list)

def create_multimodel_bracs_7class_dataloaders(model_feat_dirs, splits_dir, split_idx=0,
                                               batch_size=8, random_state=42,
                                               merge_strategy='concat', cluster_k=None):
    """
    Create multi-model train, val, test DataLoaders for BRACS 7-class dataset
    Args:
        model_feat_dirs: list of str, embedding directory for each model
        splits_dir: Path to directory containing split CSV files
        split_idx: Index of the split file to use (0-4)
        batch_size: Batch size for DataLoaders
        random_state: Random seed for reproducibility
        merge_strategy: 'concat'/'sum'/'cluster'
        cluster_k: int, number of clusters (used only for cluster strategy)
    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Instantiate datasets
    train_ds = MultiModelBRACS7ClassDataset(model_feat_dirs, splits_dir, split='train', 
                                            split_idx=split_idx, merge_strategy=merge_strategy, 
                                            cluster_k=cluster_k)
    val_ds   = MultiModelBRACS7ClassDataset(model_feat_dirs, splits_dir, split='val', 
                                            split_idx=split_idx, merge_strategy=merge_strategy, 
                                            cluster_k=cluster_k)
    test_ds  = MultiModelBRACS7ClassDataset(model_feat_dirs, splits_dir, split='test', 
                                            split_idx=split_idx, merge_strategy=merge_strategy, 
                                            cluster_k=cluster_k)

    # DataLoaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=multimodel_bracs_7class_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=multimodel_bracs_7class_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=multimodel_bracs_7class_collate_fn,
                              worker_init_fn=lambda _: np.random.seed(random_state))

    return train_loader, val_loader, test_loader

# -------------------------------
# Main (for testing)
# -------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BRACS 7-class dataset')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to feature H5 files directory')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to directory containing split CSV files')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to CSV file containing slide_id and label columns')
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Index of the split file to use (0-4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    args = parser.parse_args()

    # Test 7-class classification
    train_loader, val_loader, test_loader = create_bracs_7class_dataloaders(
        feats_path=args.feats_path,
        splits_dir=args.splits_dir,
        labels_file=args.labels_file,
        split_idx=args.split_idx,
        batch_size=args.batch_size
    )

    print(f"# train batches: {len(train_loader)}")
    print(f"# val batches:   {len(val_loader)}")
    print(f"# test batches:  {len(test_loader)}")

    for feats, labs in train_loader:
        if feats is not None and labs is not None:
            print(f"Features type: {type(feats)}")
            print(f"Labels shape: {labs.shape}")
            print(f"Labels: {labs}")
            print(f"Label range: {labs.min().item()} to {labs.max().item()}")
            
            # Handle different feature formats
            if isinstance(feats, (list, tuple)):
                print(f"Number of samples in batch: {len(feats)}")
                if len(feats) > 0:
                    first_sample = feats[0]
                    print(f"First sample type: {type(first_sample)}")
                    
                    if hasattr(first_sample, 'shape'):
                        print(f"First sample shape: {first_sample.shape}")
                    elif isinstance(first_sample, (list, tuple)):
                        print(f"First sample has {len(first_sample)} elements")
                        for i, elem in enumerate(first_sample):
                            if hasattr(elem, 'shape'):
                                print(f"  Element {i} shape: {elem.shape}")
                            else:
                                print(f"  Element {i} type: {type(elem)}")
            else:
                print(f"Features is not a list/tuple: {type(feats)}")
        break 