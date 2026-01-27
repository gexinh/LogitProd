#!/usr/bin/env python3
"""
ABMIL Inference for Survival Analysis - GitHub Release Version

This script performs inference using saved best_model.pth checkpoints for each cross-validation split.
It re-runs inference on train/val/test sets and saves results for downstream fusion tasks.

Key characteristics:
  - Loads best_model.pth from training output directory for each split.
  - Performs inference on train/val/test sets with the saved model.
  - Extracts and saves slide-level features for downstream fusion tasks.
  - Supports multiple datasets with cross-validation splits.
  - Saves predictions, hazards, survival functions, risks, and labels.

Supported datasets:
  - brca, crc, blca, kirc, lusc, gbmlgg

Expected directory structure:
  <train_output_root>/<model_name>/split_<k>/
    - best_model.pth (saved model checkpoint)
  <feats_path>/
    - features_{model_name}/ (patch-level features)
  <splits_dir>/
    - splits_{split_idx}_k.csv (train/val/test splits)

Output directory structure:
  <infer_output_root>/<model_name>/split_<k>/
    - metrics.json (inference metrics)
    - best_train_preds.npy, best_val_preds.npy, test_preds.npy
    - best_train_hazards.npy, best_val_hazards.npy, test_hazards.npy
    - best_train_S.npy, best_val_S.npy, test_S.npy
    - best_train_risks.npy, best_val_risks.npy, test_risks.npy
    - best_train_labels.npy, best_val_labels.npy, test_labels.npy
    - train_features_labels.h5, val_features_labels.h5, test_features_labels.h5
  <infer_output_root>/<model_name>/
    - summary.json (aggregated results across splits)

Usage example:
  python infer_abmil_Survival_analysis.py --dataset brca \
      --feats_path /path/to/features_{MODEL_NAME} \
      --splits_dir /path/to/splits \
      --train_output_root /path/to/train_outputs \
      --infer_output_root /path/to/infer_outputs
"""

import os
import sys
import json
import gc
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import h5py

from sksurv.metrics import concordance_index_censored

# Add datasets directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, '../../datasets'))
if datasets_dir not in sys.path:
    sys.path.insert(0, datasets_dir)

# -------------------------------
# Dataset imports
# -------------------------------
try:
    from brca_survival_dataset import create_brca_survival_dataloaders
    BRCA_SURVIVAL_AVAILABLE = True
except ImportError:
    BRCA_SURVIVAL_AVAILABLE = False
    print("Warning: brca_survival_dataset not available")

try:
    from crc_survival_dataset import create_crc_survival_dataloaders
    CRC_SURVIVAL_AVAILABLE = True
except ImportError:
    CRC_SURVIVAL_AVAILABLE = False
    print("Warning: crc_survival_dataset not available")

try:
    from blca_survival_dataset import create_blca_survival_dataloaders
    BLCA_SURVIVAL_AVAILABLE = True
except ImportError:
    BLCA_SURVIVAL_AVAILABLE = False
    print("Warning: blca_survival_dataset not available")

try:
    from kirc_survival_dataset import create_kirc_survival_dataloaders
    KIRC_SURVIVAL_AVAILABLE = True
except ImportError:
    KIRC_SURVIVAL_AVAILABLE = False
    print("Warning: kirc_survival_dataset not available")

try:
    from lusc_survival_dataset import create_lusc_survival_dataloaders
    LUSC_SURVIVAL_AVAILABLE = True
except ImportError:
    LUSC_SURVIVAL_AVAILABLE = False
    print("Warning: lusc_survival_dataset not available")

try:
    from gbmlgg_survival_dataset import create_gbmlgg_survival_dataloaders
    GBMLGG_SURVIVAL_AVAILABLE = True
except ImportError:
    GBMLGG_SURVIVAL_AVAILABLE = False
    print("Warning: gbmlgg_survival_dataset not available")

# -------------------------------
# Model import
# -------------------------------
# Get the models directory (two levels up from scripts subfolder)
model_dir = os.path.abspath(os.path.join(script_dir, '../../models'))
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

try:
    from model_abmil_with_feat import AttentionGatedWithFeat
except ImportError as e:
    raise RuntimeError(f"Cannot import AttentionGatedWithFeat: {e}")

# Full set of model hyperparameters (do not remove)
MODEL_HYPERPARAMS = {
    "uni_v2":       {"dim": 1536},
    "conch_v15":    {"dim": 768},
    "virchow2":     {"dim": 2560},
    "phikon_v2":    {"dim": 1024},
    "gigapath":     {"dim": 1536},
    "hoptimus1":    {"dim": 1536},
    "kaiko-vitl14": {"dim": 1024},
    "lunit-vits8":  {"dim": 384},
    "ctranspath":   {"dim": 768}
}

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------
# Survival analysis utilities
# -------------------------------
def concordance_index(event_times, event_observed, predicted_risk):
    """
    Wrapper that calls sksurv.metrics.concordance_index_censored.
    
    Args:
        event_times: array-like (N,) - survival times
        event_observed: array-like (N,) where 1 means event observed, 0 means censored
        predicted_risk: array-like (N,) risk scores (higher -> more risk)
    
    Returns:
        float or None: c-index (concordance index)
    """
    if event_times.size == 0:
        return None
    times = np.asarray(event_times).ravel()
    events = np.asarray(event_observed).ravel().astype(bool)
    risks = np.asarray(predicted_risk).ravel()
    try:
        res = concordance_index_censored(events, times, risks)
        return float(res[0])
    except Exception:
        return None

# -------------------------------
# Model
# -------------------------------
class SurvivalModel(nn.Module):
    """
    Survival analysis model using Attention-Based Multiple Instance Learning (ABMIL).
    Wraps AttentionGatedWithFeat for slide-level survival prediction.
    """
    def __init__(self, input_feature_dim=1024, n_classes=4, act='relu', bias=False, dropout=False):
        super().__init__()
        class Args:
            def __init__(self, input_dim, n_classes):
                self.input_dim = input_dim
                self.n_classes = n_classes
        args = Args(input_feature_dim, n_classes)
        self.abmil = AttentionGatedWithFeat(
            input_dim=input_feature_dim,
            act=act,
            bias=bias,
            dropout=dropout,
            args=args
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: either features tensor or dict {'features': tensor}
        
        Returns:
            hazards: (batch, T) - hazard probabilities
            S: (batch, T) or None - survival function
            Y_hat: optional logits
            wsi_feature: (batch, feat_dim) - slide-level features
            pred: raw logits
        """
        if isinstance(x, dict):
            features = x['features']
        else:
            features = x
        hazards, S, Y_hat, wsi_feature, pred = self.abmil(features)
        return hazards, S, Y_hat, wsi_feature, pred

# -------------------------------
# Data collection and feature extraction
# -------------------------------
def collect_preds_hazards(model, data_loader, device):
    """
    Run model over data_loader and collect predictions, hazards, survival functions, and risks.
    
    Args:
        model: SurvivalModel instance
        data_loader: DataLoader
        device: torch device
    
    Returns:
        tuple: (preds_array, hazards_array, S_array, risks_array, labels_array, c_index)
    """
    model.eval()
    preds = []
    hazards_list = []
    S_list = []
    risks = []
    labels_triplets = []  # list of [Y, censorship, survival_times]
    times_for_c = []
    events_for_c = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Collecting preds/hazards", leave=False)
        for features_list, lab in pbar:
            if features_list is None or lab is None:
                continue
            if not isinstance(lab, dict):
                continue
            batch_size = len(features_list)
            for i in range(batch_size):
                feats = features_list[i].to(device)
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)
                hazards_t, S_t, Y_hat, wsi_feature, pred = model(feats)
                # Detach cpu numpy
                pred_np = pred.squeeze(0).cpu().numpy()
                hazards_np = hazards_t.squeeze(0).cpu().numpy()
                if S_t is not None:
                    S_np = S_t.squeeze(0).cpu().numpy()
                else:
                    # Compute S from hazards if missing
                    S_np = np.cumprod(1.0 - hazards_np, axis=1)

                risk_np = -np.sum(S_np)

                preds.append(pred_np)
                hazards_list.append(hazards_np)
                S_list.append(S_np)
                risks.append(risk_np)

                # Build label triplet [Y, censorship, survival_times] with np.nan if missing
                y_val = np.nan
                c_val = np.nan
                st_val = np.nan
                if 'Y' in lab:
                    try:
                        y_val = float(lab['Y'][i].cpu().numpy())
                    except Exception:
                        try:
                            y_val = float(lab['Y'][i].item())
                        except Exception:
                            y_val = np.nan
                if 'censorship' in lab:
                    try:
                        c_val = float(lab['censorship'][i].cpu().numpy())
                    except Exception:
                        try:
                            c_val = float(lab['censorship'][i].item())
                        except Exception:
                            c_val = np.nan
                if 'survival_times' in lab:
                    try:
                        st_val = float(lab['survival_times'][i].cpu().numpy())
                    except Exception:
                        try:
                            st_val = float(lab['survival_times'][i].item())
                        except Exception:
                            st_val = np.nan

                labels_triplets.append([y_val, c_val, st_val])

                # Collect for c-index if both time and censorship present
                if not np.isnan(st_val) and not np.isnan(c_val):
                    times_for_c.append(float(st_val))
                    events_for_c.append(float(1.0 - c_val))

    if len(preds) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None)

    preds_array = np.stack(preds)
    hazards_array = np.stack(hazards_list)
    S_array = np.stack(S_list)
    risks_array = np.array(risks)
    labels_array = np.array(labels_triplets, dtype=float)

    # Prepare arrays for c-index calculation (only entries with time and event not None)
    times_valid = []
    events_valid = []
    risks_valid = []
    for t, e, r in zip(times_for_c, events_for_c, risks_array[:len(times_for_c)]):
        times_valid.append(float(t))
        events_valid.append(float(e))
        risks_valid.append(float(r))

    if len(times_valid) == 0:
        cidx = None
    else:
        try:
            cidx = concordance_index(np.array(times_valid), np.array(events_valid), np.array(risks_valid))
        except Exception:
            cidx = None

    return preds_array, hazards_array, S_array, risks_array, labels_array, cidx

def extract_features(model, data_loader, device, set_name):
    """
    Extract WSI-level features from the model.
    
    Args:
        model: SurvivalModel instance
        data_loader: DataLoader
        device: torch device
        set_name: str - name of the set (for logging)
    
    Returns:
        tuple: (wsi_features, labels, outputs)
    """
    model.eval()
    all_wsi_features = []
    all_outputs = []
    all_labels = []
    print(f"Extracting features for {set_name} set...")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Extracting {set_name}", leave=False)
        for features_list, labels in pbar:
            if features_list is None or labels is None:
                continue
            if not isinstance(labels, dict):
                continue

            batch_size = len(features_list)
            for i in range(batch_size):
                feats = features_list[i].to(device)
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)
                hazards, S, Y_hat, wsi_feature, pred = model(feats)

                all_wsi_features.append(wsi_feature.squeeze(0).cpu().numpy() if wsi_feature is not None else np.zeros(1))

                # Save raw logits pred as outputs
                out = pred.squeeze(0).cpu().numpy()
                all_outputs.append(out)
                
                # Build label triplet [Y, censorship, survival_times]
                y_val = np.nan
                c_val = np.nan
                st_val = np.nan
                if 'Y' in labels:
                    try:
                        y_val = float(labels['Y'][i].cpu().numpy())
                    except Exception:
                        try:
                            y_val = float(labels['Y'][i].item())
                        except Exception:
                            y_val = np.nan
                if 'censorship' in labels:
                    try:
                        c_val = float(labels['censorship'][i].cpu().numpy())
                    except Exception:
                        try:
                            c_val = float(labels['censorship'][i].item())
                        except Exception:
                            c_val = np.nan
                if 'survival_times' in labels:
                    try:
                        st_val = float(labels['survival_times'][i].cpu().numpy())
                    except Exception:
                        try:
                            st_val = float(labels['survival_times'][i].item())
                        except Exception:
                            st_val = np.nan
                all_labels.append([y_val, c_val, st_val])

    if len(all_wsi_features) == 0:
        print(f"Warning: No valid data found for {set_name} set")
        return None, None, None

    all_wsi_features = np.stack(all_wsi_features)
    try:
        all_outputs = np.stack(all_outputs)
    except Exception:
        all_outputs = np.array(all_outputs, dtype=object)
    all_labels = np.array(all_labels, dtype=float)
    print(f"  Features shape: {all_wsi_features.shape}")
    print(f"  Labels shape: {all_labels.shape}")
    print(f"  Outputs shape: {all_outputs.shape}")
    return all_wsi_features, all_labels, all_outputs

def save_features_to_h5(features, labels, outputs, save_path, set_name):
    """
    Save features, labels, and outputs to HDF5 file.
    
    Args:
        features: np.ndarray - WSI-level features
        labels: np.ndarray - labels
        outputs: np.ndarray - model outputs
        save_path: str - path to save file
        set_name: str - name of the set (for logging)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('labels', data=labels)
        f.create_dataset('outputs', data=outputs)
    print(f"✓ Saved {set_name} features to: {save_path}")

# -------------------------------
# DataLoader utilities
# -------------------------------
def worker_init_fn(worker_id):
    """Initialize worker with seed for reproducibility."""
    seed = SEED + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_dataloaders(dataset_type, feats_path, splits_dir, split_idx, batch_size):
    """
    Build DataLoaders for train/val/test sets.
    
    Args:
        dataset_type: str - dataset type
        feats_path: str - path to features directory
        splits_dir: str - path to splits directory
        split_idx: int - split index
        batch_size: int - batch size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create data loaders based on dataset type
    if dataset_type == 'brca':
        if not BRCA_SURVIVAL_AVAILABLE:
            raise RuntimeError("brca_survival_dataset not available")
        train_loader, val_loader, test_loader = create_brca_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'crc':
        if not CRC_SURVIVAL_AVAILABLE:
            raise RuntimeError("crc_survival_dataset not available")
        train_loader, val_loader, test_loader = create_crc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'blca':
        if not BLCA_SURVIVAL_AVAILABLE:
            raise RuntimeError("blca_survival_dataset not available")
        train_loader, val_loader, test_loader = create_blca_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'kirc':
        if not KIRC_SURVIVAL_AVAILABLE:
            raise RuntimeError("kirc_survival_dataset not available")
        train_loader, val_loader, test_loader = create_kirc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'lusc':
        if not LUSC_SURVIVAL_AVAILABLE:
            raise RuntimeError("lusc_survival_dataset not available")
        train_loader, val_loader, test_loader = create_lusc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'gbmlgg':
        if not GBMLGG_SURVIVAL_AVAILABLE:
            raise RuntimeError("gbmlgg_survival_dataset not available")
        train_loader, val_loader, test_loader = create_gbmlgg_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def _wrap(loader):
        """Wrap DataLoader with consistent settings for inference."""
        collate = loader.collate_fn if hasattr(loader, 'collate_fn') else None
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for inference
            num_workers=2,
            pin_memory=False,  # Disable pin_memory to save memory
            collate_fn=collate,
            worker_init_fn=worker_init_fn
        )

    return _wrap(train_loader), _wrap(val_loader), _wrap(test_loader)

# -------------------------------
# Utility functions
# -------------------------------
def _labels_to_serializable_list(arr):
    """
    Convert labels array (N,3) to list-of-lists replacing np.nan with None for JSON.
    If arr is 1D or scalar, convert accordingly.
    """
    if arr is None:
        return []
    a = np.asarray(arr)
    if a.size == 0:
        return []
    if a.ndim == 1:
        out = []
        for v in a:
            if isinstance(v, (list, tuple, np.ndarray)):
                # Nested? convert recursively
                out.append(_labels_to_serializable_list(np.asarray(v)))
            else:
                try:
                    if np.isnan(v):
                        out.append(None)
                    else:
                        out.append(float(v))
                except Exception:
                    out.append(None)
        return out
    elif a.ndim == 2:
        out = []
        for row in a:
            row_list = []
            for v in row:
                try:
                    if np.isnan(v):
                        row_list.append(None)
                    else:
                        row_list.append(float(v))
                except Exception:
                    row_list.append(None)
            out.append(row_list)
        return out
    else:
        # Flatten then handle
        flat = a.reshape(a.shape[0], -1)
        return _labels_to_serializable_list(flat)

def safe_mean(values):
    """Safely compute mean of values, ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))

def safe_std(values):
    """Safely compute standard deviation of values, ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.std(vals))

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='ABMIL Inference Script for Survival Analysis')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['brca', 'crc', 'blca', 'kirc', 'lusc', 'gbmlgg'],
                        help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to patch-level features directory. Supports {MODEL_NAME} placeholder.')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to splits directory containing CSV files.')
    parser.add_argument('--train_output_root', type=str, required=True,
                        help='Root directory of training outputs (contains best_model.pth for each split).')
    parser.add_argument('--infer_output_root', type=str, required=True,
                        help='Root directory for inference outputs.')
    args = parser.parse_args()

    DATASET_TYPE = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    splits_dir = args.splits_dir
    train_output_root = args.train_output_root
    infer_output_root = args.infer_output_root

    # Determine number of classes (fixed to 4 for survival analysis)
    n_classes = 4

    for MODEL_NAME, model_params in MODEL_HYPERPARAMS.items():
        input_feature_dim = model_params["dim"]
        
        # Support {MODEL_NAME} placeholder in feats_path
        if '{MODEL_NAME}' in args.feats_path:
            feats_path = args.feats_path.format(MODEL_NAME=MODEL_NAME)
        else:
            # If no placeholder, append features_{MODEL_NAME} to the path
            feats_path = os.path.join(args.feats_path, f"features_{MODEL_NAME}")

        if not os.path.exists(feats_path):
            print(f"[skip] Features directory does not exist: {feats_path}")
            continue

        output_base_dir_train = os.path.join(train_output_root, MODEL_NAME)
        if not os.path.exists(output_base_dir_train):
            print(f"[skip] Train output directory not found: {output_base_dir_train}")
            continue

        output_base_dir_infer = os.path.join(infer_output_root, MODEL_NAME)
        os.makedirs(output_base_dir_infer, exist_ok=True)

        print(f"\n===== Re-infer model {MODEL_NAME} on {DATASET_TYPE} ({args.num_splits} folds) =====")
        all_results = []

        for split_idx in range(args.num_splits):
            # Reset random seed for each split (consistent with training script)
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            split_output_dir_train = os.path.join(output_base_dir_train, f"split_{split_idx}")
            split_output_dir_infer = os.path.join(output_base_dir_infer, f"split_{split_idx}")
            os.makedirs(split_output_dir_infer, exist_ok=True)

            best_model_path = os.path.join(split_output_dir_train, "best_model.pth")
            if not os.path.exists(best_model_path):
                print(f"[skip] best_model.pth not found: {best_model_path}")
                continue

            print(f"\n--- Split {split_idx}/{args.num_splits-1} ---")

            try:
                train_loader, val_loader, test_loader = build_dataloaders(
                    DATASET_TYPE, feats_path, splits_dir, split_idx, args.batch_size
                )
            except Exception as e:
                print(f"[skip] Failed to build DataLoader ({MODEL_NAME}, split {split_idx}): {e}")
                continue

            # Build and load model
            model = SurvivalModel(
                input_feature_dim=input_feature_dim,
                n_classes=n_classes,
                act='relu',
                bias=False,
                dropout=True
            ).to(device)

            # Load best model
            best_model_state = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(best_model_state)
            model.to(device)
            print(f"Using best model from: {best_model_path}")

            # Collect predictions, hazards, risks, and labels
            train_preds, train_hazards, train_S, train_risks, train_labels, train_cidx = collect_preds_hazards(model, train_loader, device)
            val_preds, val_hazards, val_S, val_risks, val_labels, val_cidx = collect_preds_hazards(model, val_loader, device)
            test_preds, test_hazards, test_S, test_risks, test_labels, test_cidx = collect_preds_hazards(model, test_loader, device)

            def _maybe_save(prefix, preds, hazards, S, risks, labels):
                """Save numpy arrays if not empty."""
                if preds.size == 0:
                    return
                np.save(os.path.join(split_output_dir_infer, f"{prefix}_preds.npy"), preds)
                np.save(os.path.join(split_output_dir_infer, f"{prefix}_hazards.npy"), hazards)
                np.save(os.path.join(split_output_dir_infer, f"{prefix}_S.npy"), S)
                np.save(os.path.join(split_output_dir_infer, f"{prefix}_risks.npy"), risks)
                np.save(os.path.join(split_output_dir_infer, f"{prefix}_labels.npy"), labels)

            # Save inference results
            _maybe_save("best_train", train_preds, train_hazards, train_S, train_risks, train_labels)
            _maybe_save("best_val", val_preds, val_hazards, val_S, val_risks, val_labels)
            _maybe_save("test", test_preds, test_hazards, test_S, test_risks, test_labels)

            # Convert labels to serializable format
            test_labels_list = _labels_to_serializable_list(test_labels)
            train_labels_list = _labels_to_serializable_list(train_labels)
            val_labels_list = _labels_to_serializable_list(val_labels)

            # Save data for results dict (before deleting large arrays)
            split_result_data = {
                'test_preds': test_preds.tolist() if isinstance(test_preds, np.ndarray) and test_preds.size > 0 else [],
                'test_risks': test_risks.tolist() if isinstance(test_risks, np.ndarray) and test_risks.size > 0 else [],
                'test_hazards_shape': list(test_hazards.shape) if isinstance(test_hazards, np.ndarray) and test_hazards.size > 0 else None,
                'test_S_shape': list(test_S.shape) if isinstance(test_S, np.ndarray) and test_S.size > 0 else None,
                'test_labels': test_labels_list,
                'best_train_preds': train_preds.tolist() if isinstance(train_preds, np.ndarray) and train_preds.size > 0 else [],
                'best_train_risks': train_risks.tolist() if isinstance(train_risks, np.ndarray) and train_risks.size > 0 else [],
                'best_train_labels': train_labels_list,
                'best_val_preds': val_preds.tolist() if isinstance(val_preds, np.ndarray) and val_preds.size > 0 else [],
                'best_val_risks': val_risks.tolist() if isinstance(val_risks, np.ndarray) and val_risks.size > 0 else [],
                'best_val_labels': val_labels_list,
            }

            # Release memory after saving
            del train_preds, train_hazards, train_S, train_risks, train_labels
            del val_preds, val_hazards, val_S, val_risks, val_labels
            del test_preds, test_hazards, test_S, test_risks, test_labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Feature extraction
            print("  Extracting training set features...")
            train_features, train_labels_h5, train_outputs_h5 = extract_features(model, train_loader, device, "training")
            if train_features is not None:
                save_features_to_h5(train_features, train_labels_h5, train_outputs_h5,
                                    os.path.join(split_output_dir_infer, 'train_features_labels.h5'), "training")
                del train_features, train_labels_h5, train_outputs_h5
            gc.collect()

            print("  Extracting validation set features...")
            val_features, val_labels_h5, val_outputs_h5 = extract_features(model, val_loader, device, "validation")
            if val_features is not None:
                save_features_to_h5(val_features, val_labels_h5, val_outputs_h5,
                                    os.path.join(split_output_dir_infer, 'val_features_labels.h5'), "validation")
                del val_features, val_labels_h5, val_outputs_h5
            gc.collect()

            print("  Extracting test set features...")
            test_features, test_labels_h5, test_outputs_h5 = extract_features(model, test_loader, device, "test")
            if test_features is not None:
                save_features_to_h5(test_features, test_labels_h5, test_outputs_h5,
                                    os.path.join(split_output_dir_infer, 'test_features_labels.h5'), "test")
                del test_features, test_labels_h5, test_outputs_h5
            gc.collect()

            # Build results dict (consistent with training script format)
            split_result = {
                'model_name': MODEL_NAME,
                'split_idx': split_idx,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'best_model_path': best_model_path,
                'train_cindex': float(train_cidx) if train_cidx is not None else None,
                'val_cindex': float(val_cidx) if val_cidx is not None else None,
                'test_cindex': float(test_cidx) if test_cidx is not None else None,
                'best_train_cindex': float(train_cidx) if train_cidx is not None else None,
                'best_val_cindex_for_saved_model': float(val_cidx) if val_cidx is not None else None,
                **split_result_data
            }
            all_results.append(split_result)

            # Save metrics JSON (consistent with training script format)
            metrics_path = os.path.join(split_output_dir_infer, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(split_result, f, indent=4)

            # Clean up DataLoader and model
            del train_loader, val_loader, test_loader
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Split {split_idx} completed, memory cleared")

        if not all_results:
            print(f"[warning] No valid split results for model {MODEL_NAME}")
            continue

        # Save summary (consistent with training script format)
        summary = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'dataset': DATASET_TYPE,
            'model_name': MODEL_NAME,
            'num_splits_completed': len(all_results),
            'average_results': {
                'avg_best_train_cindex': safe_mean([r['train_cindex'] for r in all_results]),
                'std_train_cindex': safe_std([r['train_cindex'] for r in all_results]),
                'avg_best_val_cindex': safe_mean([r['val_cindex'] for r in all_results]),
                'std_val_cindex': safe_std([r['val_cindex'] for r in all_results]),
                'avg_test_cindex': safe_mean([r['test_cindex'] for r in all_results]),
                'std_test_cindex': safe_std([r['test_cindex'] for r in all_results]),
            },
            'individual_results': all_results
        }
        summary_path = os.path.join(output_base_dir_infer, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Also save results.json (consistent with training script)
        results_path = os.path.join(output_base_dir_infer, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSummary for model {MODEL_NAME}:")
        avg_train = summary['average_results']['avg_best_train_cindex']
        std_train = summary['average_results']['std_train_cindex']
        avg_val = summary['average_results']['avg_best_val_cindex']
        std_val = summary['average_results']['std_val_cindex']
        avg_test = summary['average_results']['avg_test_cindex']
        std_test = summary['average_results']['std_test_cindex']
        if avg_train is not None:
            print(f"  Avg train C-Index: {avg_train:.4f} ± {std_train:.4f}" if std_train is not None else f"  Avg train C-Index: {avg_train:.4f}")
        if avg_val is not None:
            print(f"  Avg val C-Index: {avg_val:.4f} ± {std_val:.4f}" if std_val is not None else f"  Avg val C-Index: {avg_val:.4f}")
        if avg_test is not None:
            print(f"  Avg test C-Index: {avg_test:.4f} ± {std_test:.4f}" if std_test is not None else f"  Avg test C-Index: {avg_test:.4f}")
        print(f"Results saved to: {output_base_dir_infer}")

    print(f"\n✓ All inference completed! Results saved to: {infer_output_root}")

if __name__ == '__main__':
    main()
