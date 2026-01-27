#!/usr/bin/env python3
"""
ABMIL Training for Survival Analysis - GitHub Release Version

This script trains an Attention-Based Multiple Instance Learning (ABMIL) model
on whole slide images (WSI) for survival analysis tasks. It runs cross-validation
splits and saves slide-level features for downstream fusion tasks.

Key characteristics:
  - Trains ABMIL model on patch-level features extracted from pre-trained models.
  - Performs 5-fold cross-validation for survival prediction.
  - Uses discrete-time survival analysis with negative log-likelihood (NLL) loss.
  - Evaluates models using concordance index (C-index).
  - Saves best model's validation and test results (hazards, survival functions, risks, labels).
  - Extracts and saves slide-level features for train/val/test sets.

Supported datasets:
  - brca, crc, blca, kirc, lusc, gbmlgg

Expected directory structure:
  <feats_path>/
    - features_{model_name}/ (patch-level features)
  <splits_dir>/
    - splits_{split_idx}_k.csv (train/val/test splits)
    - For gbmlgg: TCGA_GBMLGG_Splits_merged.csv and TCGA_GBMLGG_survival_100/splits_*.csv

Output directory structure:
  <output_base_dir>/<model_name>/split_{split_idx}/
    - best_model.pth (saved model checkpoint)
    - metrics.json (training metrics)
    - train_features_labels.h5 (slide-level features for training set)
    - val_features_labels.h5 (slide-level features for validation set)
    - test_features_labels.h5 (slide-level features for test set)
    - best_train_preds.npy, best_val_preds.npy, test_preds.npy
    - best_train_hazards.npy, best_val_hazards.npy, test_hazards.npy
    - best_train_S.npy, best_val_S.npy, test_S.npy
    - best_train_risks.npy, best_val_risks.npy, test_risks.npy
    - best_train_labels.npy, best_val_labels.npy, test_labels.npy

Usage example:
  python train_abmil_Survival_analysis.py --dataset brca \
      --feats_path /path/to/features_{MODEL_NAME} \
      --splits_dir /path/to/splits \
      --base_output_dir /path/to/outputs
  
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py

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
    print(f"Warning: cannot import AttentionGatedWithFeat: {e}")
    AttentionGatedWithFeat = None

# Full set of model hyperparameters (do not remove)
MODEL_HYPERPARAMS = {
    "uni_v2":       {"dim": 1536, "epochs": 50},
    "conch_v15":    {"dim": 768,  "epochs": 50},
    "virchow2":     {"dim": 2560, "epochs": 50},
    "phikon_v2":    {"dim": 1024, "epochs": 50},
    "gigapath":     {"dim": 1536, "epochs": 50},
    "hoptimus1":    {"dim": 1536, "epochs": 50},
    "kaiko-vitl14": {"dim": 1024, "epochs": 50},
    "lunit-vits8":  {"dim": 384,  "epochs": 50},
    "ctranspath":   {"dim": 768,  "epochs": 50}
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
from sksurv.metrics import concordance_index_censored

def concordance_index(event_times, event_observed, predicted_risk):
    """
    Wrapper that calls sksurv.metrics.concordance_index_censored.
    
    Args:
        event_times: array-like (N,) - survival times
        event_observed: array-like (N,) where 1 means event observed, 0 means censored
        predicted_risk: array-like (N,) risk scores (higher -> more risk)
    
    Returns:
        float: c-index (concordance index)
    """
    times = np.asarray(event_times).ravel()
    events = np.asarray(event_observed).ravel().astype(bool)  # True = event observed
    risks = np.asarray(predicted_risk).ravel()
    res = concordance_index_censored(events, times, risks)
    return float(res[0])

def outputs_to_risk(outputs):
    """
    Convert model 'outputs' to a 1D risk score array of length N.
    Keeps compatibility with earlier code: if outputs are (N, T) logits, sum across T.
    
    Args:
        outputs: array-like, can be (N, T) or (N,) or other shapes
    
    Returns:
        np.ndarray: 1D risk score array
    """
    if outputs is None:
        return np.array([])
    arr = np.asarray(outputs, dtype=object)
    if arr.size == 0:
        return np.array([])
    if isinstance(outputs, np.ndarray) and outputs.dtype != object:
        if outputs.ndim == 2:
            return outputs.sum(axis=1)
        elif outputs.ndim == 1:
            return outputs
        else:
            reshaped = outputs.reshape((outputs.shape[0], -1))
            return reshaped.sum(axis=1)
    try:
        stacked = np.stack(list(arr))
        if stacked.ndim == 2:
            return stacked.sum(axis=1)
        elif stacked.ndim == 1:
            return stacked
        else:
            reshaped = stacked.reshape((stacked.shape[0], -1))
            return reshaped.sum(axis=1)
    except Exception:
        out = []
        for el in arr:
            try:
                a = np.asarray(el)
                if a.ndim == 0:
                    out.append(float(a))
                elif a.ndim == 1:
                    out.append(a.sum())
                else:
                    out.append(a.reshape((a.shape[0], -1)).sum())
            except Exception:
                out.append(np.nan)
        return np.asarray(out)

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Discrete-time negative log-likelihood (NLL) loss for survival analysis.
    
    Args:
        hazards: (B, T) tensor - hazard probabilities in (0,1)
        S: (B, T) tensor or None - survival function
        Y: (B,1) or (B,) tensor of integer bin indices (1-based: 1..T)
        c: (B,1) or (B,) censorship (1=censored, 0=event)
        alpha: float - weight for uncensored loss
        eps: float - small value for numerical stability
    
    Returns:
        torch.Tensor: scalar loss value
    """
    device = hazards.device
    batch_size = hazards.size(0)
    T = hazards.size(1)

    Y = Y.view(batch_size, 1).long().to(device)
    c = c.view(batch_size, 1).float().to(device)

    if S is None:
        hazards_clamped = hazards.clamp(min=eps, max=1.0 - eps)
        S = torch.cumprod(1.0 - hazards_clamped, dim=1)

    # Defensive: if Y looks 0-based, convert to 1-based
    if torch.min(Y) == 0:
        Y = (Y + 1).clamp(min=1)

    # Clamp Y to valid range
    Y = Y.clamp(min=1, max=T)

    ones_col = torch.ones((batch_size, 1), dtype=S.dtype, device=device)
    S_padded = torch.cat([ones_col, S], dim=1)  # (B, T+1)

    hazards_idx = (Y - 1).clamp(min=0, max=T-1)
    hazards_at_Y = torch.gather(hazards, 1, hazards_idx)  # (B,1)

    S_before = torch.gather(S_padded, 1, (Y - 1).clamp(min=0))  # S_{Y-1}
    S_at_Y = torch.gather(S_padded, 1, Y.clamp(max=T))          # S_{Y}

    uncensored_loss = - (1.0 - c) * (torch.log(S_before.clamp(min=eps)) + torch.log(hazards_at_Y.clamp(min=eps)))
    censored_loss = - c * torch.log(S_at_Y.clamp(min=eps))

    neg_l = censored_loss + uncensored_loss
    loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

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
        if AttentionGatedWithFeat is None:
            raise RuntimeError("AttentionGatedWithFeat not available; check model imports.")
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
# Training and validation
# -------------------------------
def train_epoch(model, train_loader, optimizer, device, alpha=0.4):
    """
    Train for one epoch.
    
    Args:
        model: SurvivalModel instance
        train_loader: DataLoader for training
        optimizer: optimizer
        device: torch device
        alpha: float - weight for uncensored loss
    
    Returns:
        tuple: (avg_loss, avg_cindex)
    """
    model.train()
    total_loss = 0.0
    cindex_sum = 0.0
    cindex_count = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features_list, labels in pbar:
        if features_list is None or labels is None:
            continue
        if not isinstance(labels, dict):
            continue

        has_Y = 'Y' in labels
        has_surv_times = 'survival_times' in labels
        has_c = 'censorship' in labels

        if not has_c or (not has_Y and not has_surv_times):
            continue

        batch_size = len(features_list)
        optimizer.zero_grad()

        batch_hazards = []
        batch_S = []
        Y_list = []
        c_list = []
        times_list = []

        for i in range(batch_size):
            feats = features_list[i].to(device)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)  # add batch dim

            hazards, S, Y_hat, wsi_feature, _pred = model(feats)  # hazards shape (1, T)
            T = hazards.size(1)

            # Prefer Y from dataset (expected 1-based). If not present, map survival_times -> 1..T
            Yi = labels['Y'][i].to(device).long().view(1)
            Yi = Yi.clamp(min=1, max=T)

            ci = labels['censorship'][i].to(device).float().view(1)
            batch_hazards.append(hazards)    # list of (1,T)
            if S is not None:
                batch_S.append(S)
            else:
                batch_S.append(torch.cumprod(1 - hazards, dim=1))
            Y_list.append(Yi)
            c_list.append(ci)

            # For c-index later we want risk (aggregate)
            times_list.append(labels.get('survival_times', Yi.float())[i].to(device).float())

        hazards_batch = torch.cat(batch_hazards, dim=0)  # (B, T)
        S_batch = torch.cat(batch_S, dim=0)              # (B, T)
        Y_batch = torch.cat(Y_list, dim=0).view(batch_size, 1)  # (B,1) 1-based expected
        c_batch = torch.cat(c_list, dim=0).view(batch_size, 1)
        times_batch = torch.stack(times_list).view(batch_size)

        # Use NLL loss
        loss = nll_loss(hazards=hazards_batch, S=S_batch, Y=Y_batch, c=c_batch, alpha=alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute batch c-index
        risk = (-torch.sum(S_batch, dim=1)).detach()
        try:
            # Convert to numpy arrays for concordance_index wrapper
            times_np = times_batch.detach().cpu().numpy()
            events_np = (1.0 - c_batch.view(-1)).detach().cpu().numpy()  # 1 = observed event
            risks_np = risk.detach().cpu().numpy()
            cidx = concordance_index(times_np, events_np, risks_np)
        except Exception:
            cidx = 0.0
        cindex_sum += cidx
        cindex_count += 1

    avg_loss = total_loss / max(1, len(train_loader))
    avg_cindex = cindex_sum / max(1, cindex_count) if cindex_count > 0 else 0.0
    return avg_loss, avg_cindex

def validate(model, val_loader, device, alpha=0.4):
    """
    Validate the model.
    
    Args:
        model: SurvivalModel instance
        val_loader: DataLoader for validation
        device: torch device
        alpha: float - weight for uncensored loss
    
    Returns:
        tuple: (avg_loss, c_index, 0.0, 0.0, outputs, labels)
    """
    model.eval()
    total_loss = 0.0
    all_times = []
    all_events = []
    all_risks = []
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for features_list, labels in pbar:
            if features_list is None or labels is None:
                continue
            if not isinstance(labels, dict):
                continue

            batch_size = len(features_list)
            batch_hazards = []
            batch_S = []
            Y_list = []
            c_list = []
            times_list = []
            outs = []

            for i in range(batch_size):
                feats = features_list[i].to(device)
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)
                hazards, S, Y_hat, wsi_feature, _pred = model(feats)
                T = hazards.size(1)

                if 'Y' in labels:
                    Yi = labels['Y'][i].to(device).long().view(1)
                elif 'survival_times' in labels:
                    st = labels['survival_times'][i].to(device).float().item()
                    Yi_val = int(np.floor(st))
                    Yi = torch.tensor([Yi_val + 1], device=device, dtype=torch.long).clamp(min=1, max=T)
                else:
                    continue

                Yi = Yi.clamp(min=1, max=T)
                ci = labels['censorship'][i].to(device).float().view(1)

                batch_hazards.append(hazards)
                batch_S.append(S if S is not None else torch.cumprod(1 - hazards, dim=1))
                Y_list.append(Yi)
                c_list.append(ci)
                tval = labels.get('survival_times', Yi.float())[i].to(device).float()
                times_list.append(tval)

                # Save raw logits pred as outputs (keeps compatibility)
                outs.append(_pred.squeeze(0).cpu().numpy())

                if 'survival_times' in labels:
                    all_labels.append(labels['survival_times'][i].cpu().numpy())
                elif 'Y' in labels:
                    all_labels.append(labels['Y'][i].cpu().numpy())
                else:
                    all_labels.append(None)

            if len(batch_hazards) == 0:
                continue

            hazards_batch = torch.cat(batch_hazards, dim=0)
            S_batch = torch.cat(batch_S, dim=0)
            Y_batch = torch.cat(Y_list, dim=0).view(len(batch_hazards), 1)
            c_batch = torch.cat(c_list, dim=0).view(len(batch_hazards), 1)
            times_batch = torch.stack(times_list).view(len(batch_hazards))

            loss = nll_loss(hazards=hazards_batch, S=S_batch, Y=Y_batch, c=c_batch, alpha=alpha)
            total_loss += loss.item()

            all_risks.append((-torch.sum(S_batch, dim=1)).detach().cpu().numpy())
            all_times.append(times_batch.detach().cpu().numpy())
            all_events.append((1.0 - c_batch).detach().cpu().numpy())  # 1 = observed event
            all_outputs.extend(outs)

    if len(all_risks) == 0:
        return None, None, None, None, None, None

    times = np.concatenate(all_times)
    events = np.concatenate(all_events)
    risks = np.concatenate(all_risks)
    outputs = np.stack(all_outputs) if len(all_outputs) > 0 else np.array(all_outputs)
    avg_loss = total_loss / max(1, len(val_loader))
    c_index = concordance_index(times, events, risks)
    return avg_loss, c_index, 0.0, 0.0, outputs, np.array(all_labels)

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

# -------------------------------
# Feature extraction
# -------------------------------
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

# -------------------------------
# Main training function
# -------------------------------
def train_single_gpu(split_idx, feats_path, splits_dir, output_base_dir, device,
                     input_feature_dim, num_epochs, model_name, dataset_type='brca', 
                     batch_size=8):
    """
    Train model for a single split on a single GPU.
    
    Args:
        split_idx: int - split index
        feats_path: str - path to features directory
        splits_dir: str - path to splits directory
        output_base_dir: str - base output directory
        device: torch device
        input_feature_dim: int - input feature dimension
        num_epochs: int - number of epochs
        model_name: str - model name
        dataset_type: str - dataset type
        batch_size: int - batch size
    
    Returns:
        dict or None: training results
    """
    print(f"Training split {split_idx} on device {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    split_output_dir = os.path.join(output_base_dir, f"split_{split_idx}")
    os.makedirs(split_output_dir, exist_ok=True)

    # Create data loaders based on dataset type
    if dataset_type == 'brca':
        if not BRCA_SURVIVAL_AVAILABLE:
            print("Error: brca_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_brca_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'crc':
        if not CRC_SURVIVAL_AVAILABLE:
            print("Error: crc_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_crc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'blca':
        if not BLCA_SURVIVAL_AVAILABLE:
            print("Error: blca_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_blca_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'kirc':
        if not KIRC_SURVIVAL_AVAILABLE:
            print("Error: kirc_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_kirc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'lusc':
        if not LUSC_SURVIVAL_AVAILABLE:
            print("Error: lusc_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_lusc_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    elif dataset_type == 'gbmlgg':
        if not GBMLGG_SURVIVAL_AVAILABLE:
            print("Error: gbmlgg_survival_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_gbmlgg_survival_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size, n_bins=4
        )
    else:
        print(f"Error: dataset_type {dataset_type} not supported in this script.")
        return None

    # Recreate data loaders with proper settings
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=val_loader.collate_fn if hasattr(val_loader, 'collate_fn') else None,
        worker_init_fn=worker_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=test_loader.collate_fn if hasattr(test_loader, 'collate_fn') else None,
        worker_init_fn=worker_init_fn
    )

    # Set number of classes to 4 (t=4)
    n_classes = 4
    model = SurvivalModel(
        input_feature_dim=input_feature_dim,
        n_classes=n_classes,
        act='relu',
        bias=False,
        dropout=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Use val c-index as criterion for saving best model (higher is better)
    best_val_loss = float('inf')   # keep for record
    best_val_cindex = -1e9
    best_model_state = None
    best_epoch = -1
    training_history = {'epoch': [], 'train_loss': [], 'train_cindex': [], 'val_loss': [], 'val_cindex': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_cindex = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_cindex, _, _, val_outputs, val_labels = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train C-Index: {train_cindex:.4f} | Val Loss: {val_loss if val_loss is not None else float('nan'):.4f} | Val C-Index: {val_cindex if val_cindex is not None else float('nan'):.4f}")

        if val_cindex is None:
            # Can't evaluate c-index this epoch, skip saving
            continue

        # If val_cindex improves -> save model
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_val_loss = val_loss if val_loss is not None else best_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            print(f"  ✓ Val C-Index improved to {best_val_cindex:.4f}, saving model")

        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_cindex'].append(train_cindex)
        training_history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        training_history['val_cindex'].append(val_cindex if val_cindex is not None else float('nan'))

    if best_model_state is None:
        print("No valid model saved during training.")
        return None

    model_path = os.path.join(split_output_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    print(f"✓ Saved best model from epoch {best_epoch}! Val C-Index: {best_val_cindex:.4f} (val_loss at that epoch: {best_val_loss:.4f})")

    np.save(os.path.join(split_output_dir, 'training_history.npy'), training_history)

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)
    print(f"Using best model from epoch {best_epoch} (val_cindex: {best_val_cindex:.4f})")

    # Collect and save train/val/test preds/hazards/risk
    train_preds, train_hazards, train_S, train_risks, train_labels_arr, train_cidx = collect_preds_hazards(model, train_loader, device)
    val_preds, val_hazards, val_S, val_risks, val_labels_arr, val_cidx = collect_preds_hazards(model, val_loader, device)
    test_preds, test_hazards, test_S, test_risks, test_labels_arr, test_cidx = collect_preds_hazards(model, test_loader, device)

    # If collect returned empty test_preds -> skip
    if test_preds.size == 0:
        print("Warning: test outputs None or empty — skipping saving for this split")
        return None

    # Save numpy artifacts (preds/hazards/S/risks/labels)
    np.save(os.path.join(split_output_dir, 'test_preds.npy'), test_preds)
    np.save(os.path.join(split_output_dir, 'test_hazards.npy'), test_hazards)
    np.save(os.path.join(split_output_dir, 'test_S.npy'), test_S)
    np.save(os.path.join(split_output_dir, 'test_risks.npy'), test_risks)
    np.save(os.path.join(split_output_dir, 'test_labels.npy'), test_labels_arr)

    if train_preds.size > 0:
        np.save(os.path.join(split_output_dir, 'best_train_preds.npy'), train_preds)
        np.save(os.path.join(split_output_dir, 'best_train_hazards.npy'), train_hazards)
        np.save(os.path.join(split_output_dir, 'best_train_S.npy'), train_S)
        np.save(os.path.join(split_output_dir, 'best_train_risks.npy'), train_risks)
        np.save(os.path.join(split_output_dir, 'best_train_labels.npy'), train_labels_arr)
    if val_preds.size > 0:
        np.save(os.path.join(split_output_dir, 'best_val_preds.npy'), val_preds)
        np.save(os.path.join(split_output_dir, 'best_val_hazards.npy'), val_hazards)
        np.save(os.path.join(split_output_dir, 'best_val_S.npy'), val_S)
        np.save(os.path.join(split_output_dir, 'best_val_risks.npy'), val_risks)
        np.save(os.path.join(split_output_dir, 'best_val_labels.npy'), val_labels_arr)

    # Build results dict to write JSON summary
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

    test_labels_list = _labels_to_serializable_list(test_labels_arr)
    train_labels_list = _labels_to_serializable_list(train_labels_arr)
    val_labels_list = _labels_to_serializable_list(val_labels_arr)
    results = {
        'model_name': model_name,
        'split_idx': split_idx,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_cindex': float(best_val_cindex) if best_val_cindex is not None else 0.0,
        'test_cindex': float(test_cidx) if test_cidx is not None else 0.0,
        'test_preds': test_preds.tolist() if isinstance(test_preds, np.ndarray) else test_preds,
        'test_risks': test_risks.tolist() if isinstance(test_risks, np.ndarray) else test_risks,
        'test_hazards_shape': list(test_hazards.shape) if isinstance(test_hazards, np.ndarray) else None,
        'test_S_shape': list(test_S.shape) if isinstance(test_S, np.ndarray) else None,
        'test_labels': test_labels_list,
        'model_weights_path': model_path,
        'best_train_cindex': float(train_cidx) if train_cidx is not None else 0.0,
        'best_train_preds': train_preds.tolist() if isinstance(train_preds, np.ndarray) else train_preds,
        'best_train_risks': train_risks.tolist() if isinstance(train_risks, np.ndarray) else train_risks,
        'best_train_labels': train_labels_list,
        'best_val_cindex_for_saved_model': float(val_cidx) if val_cidx is not None else 0.0,
        'best_val_preds': val_preds.tolist() if isinstance(val_preds, np.ndarray) else val_preds,
        'best_val_risks': val_risks.tolist() if isinstance(val_risks, np.ndarray) else val_risks,
        'best_val_labels': val_labels_list,
    }

    # Write metrics json
    with open(os.path.join(split_output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Feature extraction
    train_features, train_labels_h5, train_outputs_h5 = extract_features(model, train_loader, device, "training")
    if train_features is not None:
        save_features_to_h5(train_features, train_labels_h5, train_outputs_h5,
                             os.path.join(split_output_dir, 'train_features_labels.h5'), "training")
    val_features, val_labels_h5, val_outputs_h5 = extract_features(model, val_loader, device, "validation")
    if val_features is not None:
        save_features_to_h5(val_features, val_labels_h5, val_outputs_h5,
                             os.path.join(split_output_dir, 'val_features_labels.h5'), "validation")
    test_features, test_labels_h5, test_outputs_h5 = extract_features(model, test_loader, device, "test")
    if test_features is not None:
        save_features_to_h5(test_features, test_labels_h5, test_outputs_h5,
                             os.path.join(split_output_dir, 'test_features_labels.h5'), "test")

    return results

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='ABMIL Training Script for Survival Analysis (Single GPU)')
    parser.add_argument('--dataset', type=str, default='brca', 
                        choices=['brca', 'crc', 'blca', 'kirc', 'lusc', 'gbmlgg'],
                        help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to patch-level features directory. Supports {MODEL_NAME} placeholder.')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to splits directory containing CSV files.')
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help='Base output directory for saving results.')
    args = parser.parse_args()
    
    DATASET_TYPE = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (single GPU)")

    for MODEL_NAME, model_params in MODEL_HYPERPARAMS.items():
        INPUT_FEATURE_DIM = model_params["dim"]
        NUM_EPOCHS = model_params["epochs"]

        # Use provided arguments (all paths are required)
        # Support {MODEL_NAME} placeholder in feats_path
        if '{MODEL_NAME}' in args.feats_path:
            feats_path = args.feats_path.format(MODEL_NAME=MODEL_NAME)
        else:
            # If no placeholder, append features_{MODEL_NAME} to the path
            feats_path = os.path.join(args.feats_path, f"features_{MODEL_NAME}")

        splits_dir = args.splits_dir
        base_output_dir = args.base_output_dir

        print(f"\n{'='*80}")
        print(f"Training model: {MODEL_NAME} on dataset: {DATASET_TYPE}")
        print(f"Features path: {feats_path}")
        print(f"Splits dir: {splits_dir}")
        print(f"Input dim: {INPUT_FEATURE_DIM}, epochs: {NUM_EPOCHS}, batch_size: {args.batch_size}")
        print(f"{'='*80}")

        if not os.path.exists(feats_path):
            print(f"Error: features directory does not exist: {feats_path}")
            continue
        if not os.path.exists(splits_dir):
            print(f"Error: splits directory does not exist: {splits_dir}")
            continue

        output_base_dir = os.path.join(base_output_dir, MODEL_NAME)
        os.makedirs(output_base_dir, exist_ok=True)

        all_results = []
        start_time = time.time()
        num_splits = 5
        for split_idx in range(num_splits):
            print(f"\n{'-'*60}")
            print(f"Split {split_idx} / {num_splits - 1}")
            print(f"{'-'*60}")
            try:
                result = train_single_gpu(
                    split_idx, feats_path, splits_dir, output_base_dir,
                    device, INPUT_FEATURE_DIM, NUM_EPOCHS, MODEL_NAME,
                    DATASET_TYPE, args.batch_size
                )
                if result:
                    all_results.append(result)
                print(f"Split {split_idx} completed.")
            except Exception as e:
                print(f"Split {split_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_results:
            print(f"No successful splits for model {MODEL_NAME}")
            continue

        # Summary
        avg_best_val_loss = np.mean([r['best_val_loss'] for r in all_results])
        avg_best_val_cindex = np.mean([r['best_val_cindex'] for r in all_results])
        avg_test_cindex = np.mean([r['test_cindex'] for r in all_results])

        std_best_val_loss = np.std([r['best_val_loss'] for r in all_results])
        std_best_val_cindex = np.std([r['best_val_cindex'] for r in all_results])
        std_test_cindex = np.std([r['test_cindex'] for r in all_results])

        total_time = time.time() - start_time

        summary = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_time_seconds': total_time,
            'num_splits_completed': len(all_results),
            'model_name': MODEL_NAME,
            'config': {
                'input_feature_dim': INPUT_FEATURE_DIM,
                'num_epochs': NUM_EPOCHS,
                'feats_path': feats_path,
                'splits_dir': splits_dir,
                'model_name': MODEL_NAME,
                'n_bins': 4
            },
            'average_results': {
                'avg_best_val_loss': float(avg_best_val_loss),
                'avg_best_val_cindex': float(avg_best_val_cindex),
                'avg_test_cindex': float(avg_test_cindex),
                'std_best_val_loss': float(std_best_val_loss),
                'std_best_val_cindex': float(std_best_val_cindex),
                'std_test_cindex': float(std_test_cindex)
            },
            'individual_results': all_results
        }

        with open(os.path.join(output_base_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(output_base_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSummary for model {MODEL_NAME}:")
        print(f"  Avg best val C-Index: {avg_best_val_cindex:.4f} ± {std_best_val_cindex:.4f}")
        print(f"  Avg test C-Index: {avg_test_cindex:.4f} ± {std_test_cindex:.4f}")
        print(f"Results saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
