#!/usr/bin/env python3
"""
ABMIL Training with Slide-level Feature Extraction - GitHub Release Version

This script trains an Attention-Based Multiple Instance Learning (ABMIL) model
on whole slide images (WSI) for classification tasks. It runs cross-validation
splits and saves slide-level features for downstream fusion tasks.

Key characteristics:
  - Trains ABMIL model on patch-level features extracted from pre-trained models.
  - Performs 5-fold cross-validation (or single split for PANDA dataset).
  - Saves best model's validation and test results (outputs, predictions, labels, probabilities).
  - Extracts and saves slide-level features for train/val/test sets.
  - Uses softmax probabilities for AUC calculation (supports binary and multiclass).

Supported datasets:
  - crc, bracs, bracs_7class, brca, panda

Expected directory structure:
  <feats_path>/
    - features_{model_name}/ (patch-level features)
  <splits_dir>/
    - splits_{split_idx}_k.csv (train/val/test splits)

Output directory structure:
  <output_base_dir>/split_{split_idx}/
    - best_model.pth (saved model checkpoint)
    - metrics.json (training metrics)
    - train_features_labels.h5 (slide-level features for training set)
    - val_features_labels.h5 (slide-level features for validation set)
    - test_features_labels.h5 (slide-level features for test set)
    - best_train_outputs.npy, best_val_outputs.npy, test_outputs.npy
    - best_train_labels.npy, best_val_labels.npy, test_labels.npy
    - best_train_predictions.npy, best_val_predictions.npy, test_predictions.npy
    - best_train_probabilities.npy, best_val_probabilities.npy, test_probabilities.npy

Usage example:
  python train_abmil.py --dataset crc \
      --feats_path ./features \
      --splits_dir ./splits \
      --base_output_dir ./outputs
  
  # With MODEL_NAME placeholder:
  python train_abmil.py --dataset crc \
      --feats_path /path/to/features_{MODEL_NAME} \
      --splits_dir /path/to/splits \
      --base_output_dir /path/to/outputs
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import argparse
import sys
import json
import time
from datetime import datetime
import h5py

# Add datasets directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, '../../datasets'))
if datasets_dir not in sys.path:
    sys.path.insert(0, datasets_dir)

# Dataset imports
try:
    from crc_dataset import create_crc_dataloaders
    CRC_AVAILABLE = True
except ImportError:
    CRC_AVAILABLE = False
    print("Warning: crc_dataset not available")

try:
    from bracs_dataset import create_bracs_dataloaders
    BRACS_AVAILABLE = True
except ImportError:
    BRACS_AVAILABLE = False
    print("Warning: bracs_dataset not available")

try:
    from bracs_7class_dataset import create_bracs_7class_dataloaders
    BRACS_7CLASS_AVAILABLE = True
except ImportError:
    BRACS_7CLASS_AVAILABLE = False
    print("Warning: bracs_7class_dataset not available")

try:
    from brca_dataset import create_brca_dataloaders
    BRCA_AVAILABLE = True
except ImportError:
    BRCA_AVAILABLE = False
    print("Warning: brca_dataset not available")

try:
    from panda_dataset import create_panda_dataloaders
    PANDA_AVAILABLE = True
except ImportError:
    PANDA_AVAILABLE = False
    print("Warning: panda_dataset not available")

# -------------------------------
# Model implementation import
# Get the models directory (two levels up from scripts subfolder)
model_dir = os.path.abspath(os.path.join(script_dir, '../../models'))
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
from model_abmil_with_feat import AttentionGatedWithFeat

# Full set of model hyperparameters (do not remove)
MODEL_HYPERPARAMS = {
    "uni_v2":       {"dim": 1536, "epochs": 20},
    "conch_v15":    {"dim": 768,  "epochs": 20},
    "virchow2":     {"dim": 2560, "epochs": 20},
    "phikon_v2":    {"dim": 1024, "epochs": 20},
    "gigapath":     {"dim": 1536, "epochs": 20},
    "hoptimus1":    {"dim": 1536, "epochs": 20},
    "kaiko-vitl14": {"dim": 1024, "epochs": 20},
    "lunit-vits8":  {"dim": 384,  "epochs": 20},
    "ctranspath":   {"dim": 768,  "epochs": 20}
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
# Model utilities
# -------------------------------
def calculate_abmil_flops(model, input_feature_dim, num_patches=1000, batch_size=1):
    """
    Calculate FLOPs (Floating Point Operations) for ABMIL model.
    Uses a typical input size for estimation.
    """
    try:
        from thop import profile, clever_format
        # Create dummy input: [batch_size, num_patches, input_feature_dim]
        dummy_input = torch.randn(batch_size, num_patches, input_feature_dim).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # Format with appropriate precision
        if flops >= 1e9:
            flops_str = f"{flops / 1e9:.3f}G"
        elif flops >= 1e6:
            flops_str = f"{flops / 1e6:.3f}M"
        elif flops >= 1e3:
            flops_str = f"{flops / 1e3:.3f}K"
        else:
            flops_str = f"{flops:.0f}"
        
        if params >= 1e6:
            params_str = f"{params / 1e6:.3f}M"
        elif params >= 1e3:
            params_str = f"{params / 1e3:.3f}K"
        else:
            params_str = f"{params:.0f}"
        
        return flops_str, params_str, flops, params
    except ImportError:
        # Manual estimation if thop is not available
        # ABMIL attention mechanism (simplified estimation)
        total_flops = 0
        # Attention: Q, K, V projections and attention computation
        total_flops += num_patches * input_feature_dim * input_feature_dim  # attention weights (QK^T)
        total_flops += num_patches * input_feature_dim  # weighted sum (attention * V)
        # Final classification layer
        total_flops += input_feature_dim * 2  # final linear (assuming binary classification)
        
        # Format FLOPs
        if total_flops >= 1e9:
            flops_str = f"{total_flops / 1e9:.3f}G"
        elif total_flops >= 1e6:
            flops_str = f"{total_flops / 1e6:.3f}M"
        elif total_flops >= 1e3:
            flops_str = f"{total_flops / 1e3:.3f}K"
        else:
            flops_str = f"{total_flops:.0f}"
        
        params = sum(p.numel() for p in model.parameters())
        if params >= 1e6:
            params_str = f"{params / 1e6:.3f}M"
        elif params >= 1e3:
            params_str = f"{params / 1e3:.3f}K"
        else:
            params_str = f"{params:.0f}"
        
        return flops_str, params_str, total_flops, params

# -------------------------------
# Data utilities
# -------------------------------
def custom_collate_fn(batch):
    """
    Custom collate function for handling variable-length feature sequences.
    Handles both dict labels (brca) and scalar labels (other datasets).
    """
    features_list = []
    labels = []
    for item in batch:
        if len(item) == 2:
            features, label = item
        elif len(item) == 3:
            features, label, _ = item  # Some datasets return 3 items
        else:
            continue
        if features is None or label is None:
            continue
        features_list.append(features)
        # Handle both dict labels (brca) and scalar labels (other datasets)
        if isinstance(label, dict):
            # Extract numeric label from dict
            if 'label_numeric' in label:
                # For brca dataset
                labels.append(label['label_numeric'].item())
            elif 'labels' in label:
                # For multi-label datasets
                labels.append(label['labels'].item() if label['labels'].numel() == 1 else label['labels'].tolist())
            else:
                # Fallback: try to extract first numeric value
                labels.append(list(label.values())[0].item() if hasattr(list(label.values())[0], 'item') else list(label.values())[0])
        else:
            labels.append(label.item() if torch.is_tensor(label) else label)
    if len(features_list) == 0:
        return [], torch.tensor([])
    return features_list, torch.tensor(labels)

# -------------------------------
# Model
# -------------------------------
class BinaryClassificationModel(nn.Module):
    """
    Binary classification model using Attention-Based Multiple Instance Learning (ABMIL).
    Wraps AttentionGatedWithFeat for slide-level classification.
    """
    def __init__(self, input_feature_dim=1024, n_classes=1, act='relu', bias=False, dropout=False):
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
        if isinstance(x, dict):
            features = x['features']
        else:
            features = x
        pred, Y_prob, Y_hat, wsi_feature, result_dict = self.abmil(features)
        return pred, Y_prob, wsi_feature

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features_list, labels in pbar:
        if features_list is None or labels is None or len(features_list) == 0:
            continue
        labels = labels.to(device)
        batch_size = len(features_list)
        batch_outputs = []
        for i in range(batch_size):
            features = features_list[i].to(device)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            outputs, _, _ = model(features)
            batch_outputs.append(outputs)
        outputs = torch.cat(batch_outputs, dim=0)
        optimizer.zero_grad()
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        train_correct += (predicted == labels.long()).sum().item()
        train_total += labels.size(0)
    avg_loss = total_loss / max(1, len(train_loader))
    avg_acc = train_correct / max(1, train_total)
    return avg_loss, avg_acc

def compute_metrics_from_logits(all_outputs, all_labels):
    """
    Unified processing: convert logits -> probabilities and compute AUC / acc / f1.
    Handles both binary and multiclass classification.
    """
    # Ensure labels are integers for classification metrics
    all_labels = np.asarray(all_labels, dtype=np.int64).ravel()
    preds = np.argmax(all_outputs, axis=1)
    # softmax to probabilities
    probabilities = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
    # AUC
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(all_labels, probabilities[:, 1])
        # Binary classification F1: use binary mode to handle edge cases
        try:
            f1 = f1_score(all_labels, preds, zero_division=0)
        except ValueError:
            # If a class is missing in labels or predictions, use average='binary'
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
    else:
        auc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, preds)
    return auc, accuracy, f1, probabilities, preds

def validate(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for features_list, labels in pbar:
            if features_list is None or labels is None or len(features_list) == 0:
                continue
            labels = labels.to(device)
            batch_size = len(features_list)
            batch_outputs = []
            for i in range(batch_size):
                features = features_list[i].to(device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                outputs, _, _ = model(features)
                batch_outputs.append(outputs)
            outputs = torch.cat(batch_outputs, dim=0)
            loss = criterion(outputs, labels.long())
            total_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if len(all_outputs) == 0:
        return None, None, None, None, None, None
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc, accuracy, f1, probabilities, preds = compute_metrics_from_logits(all_outputs, all_labels)
    avg_loss = total_loss / max(1, len(val_loader))
    return avg_loss, auc, accuracy, f1, all_outputs, all_labels

def test_model(model, test_loader, device):
    """Test model on test set."""
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for features_list, labels in pbar:
            if features_list is None or labels is None or len(features_list) == 0:
                continue
            labels = labels.to(device)
            batch_size = len(features_list)
            batch_outputs = []
            for i in range(batch_size):
                features = features_list[i].to(device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                outputs, _, _ = model(features)
                batch_outputs.append(outputs)
            outputs = torch.cat(batch_outputs, dim=0)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if len(all_outputs) == 0:
        return None, None, None, None, None
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc, accuracy, f1, probabilities, preds = compute_metrics_from_logits(all_outputs, all_labels)
    return auc, accuracy, f1, all_outputs, all_labels

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(model, data_loader, device, set_name):
    """
    Extract slide-level features from model.
    Returns features, labels, and outputs for saving.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_outputs = []
    print(f"Extracting features for {set_name} set...")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Extracting {set_name}", leave=False)
        for features_list, labels in pbar:
            if features_list is None or labels is None or len(features_list) == 0:
                continue
            labels = labels.to(device)
            batch_size = len(features_list)
            batch_outputs = []
            batch_wsi_features = []
            for i in range(batch_size):
                features = features_list[i].to(device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                outputs, Y_prob, wsi_feature = model(features)
                batch_outputs.append(outputs)
                batch_wsi_features.append(wsi_feature.squeeze(0))
            outputs = torch.cat(batch_outputs, dim=0)
            slide_features = torch.stack(batch_wsi_features, dim=0)
            all_outputs.append(outputs.cpu().numpy())
            all_features.append(slide_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if len(all_outputs) == 0:
        print(f"Warning: No valid data found for {set_name} set")
        return None, None, None
    all_outputs = np.concatenate(all_outputs)
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    print(f"  Features shape: {all_features.shape}")
    print(f"  Labels shape: {all_labels.shape}")
    print(f"  Outputs shape: {all_outputs.shape}")
    return all_features, all_labels, all_outputs

def save_features_to_h5(features, labels, outputs, save_path, set_name):
    """Save slide-level features, labels, and outputs to HDF5 file."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('labels', data=labels)
        f.create_dataset('outputs', data=outputs)
    print(f"✓ Saved {set_name} features to: {save_path}")

# -------------------------------
# Main training function
# -------------------------------
def train_single_gpu(split_idx, feats_path, splits_dir, output_base_dir, device,
                     input_feature_dim, num_epochs, model_name, dataset_type='crc', batch_size=8):
    """
    Train ABMIL model for a single split.
    
    Args:
        split_idx: Split index for cross-validation
        feats_path: Path to patch-level features
        splits_dir: Directory containing split CSV files
        output_base_dir: Base directory for saving outputs
        device: Device to use for training
        input_feature_dim: Dimension of input features
        num_epochs: Number of training epochs
        model_name: Name of the model (for logging)
        dataset_type: Type of dataset
        batch_size: Batch size for training
    
    Returns:
        Dictionary containing training results and metrics
    """
    print(f"Training split {split_idx} on device {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    split_output_dir = os.path.join(output_base_dir, f"split_{split_idx}")
    os.makedirs(split_output_dir, exist_ok=True)

    # Data loaders
    if dataset_type == 'crc':
        if not CRC_AVAILABLE:
            print("Error: crc_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_crc_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size)

    elif dataset_type == 'bracs':
        if not BRACS_AVAILABLE:
            print("Error: bracs_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_bracs_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx,
            batch_size=batch_size, binary_classification=False)

    elif dataset_type == 'bracs_7class':
        if not BRACS_7CLASS_AVAILABLE:
            print("Error: bracs_7class_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_bracs_7class_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size)

    elif dataset_type == 'brca':
        if not BRCA_AVAILABLE:
            print("Error: brca_dataset not available")
            return None
        train_loader, val_loader, test_loader = create_brca_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, split_idx=split_idx, batch_size=batch_size)

    elif dataset_type == 'panda':
        if not PANDA_AVAILABLE:
            print("Error: panda_dataset not available")
            return None
        # PANDA currently uses only one split file splits_0_k.csv, so split_idx is ignored
        train_loader, val_loader, test_loader = create_panda_dataloaders(
            feats_path=feats_path, splits_dir=splits_dir, batch_size=batch_size)

    else:
        print(f"Error: Unknown dataset type: {dataset_type}")
        return None

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Model
    if dataset_type == 'bracs':
        n_classes = 3
    elif dataset_type == 'bracs_7class':
        n_classes = 7
    elif dataset_type == 'panda':
        # PANDA: 6-class classification
        n_classes = 6
    else:
        n_classes = 2

    model = BinaryClassificationModel(
        input_feature_dim=input_feature_dim,
        n_classes=n_classes,
        act='relu',
        bias=False,
        dropout=True
    ).to(device)

    # Calculate FLOPs and parameters
    flops_str, params_str, flops_value, params_value = calculate_abmil_flops(
        model, input_feature_dim, num_patches=1000, batch_size=1
    )
    print(f"Model FLOPs: {flops_str}, Parameters: {params_str}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    training_history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc, val_f1, val_outputs, val_labels = validate(model, val_loader, criterion, device)
        if val_outputs is None:
            continue
        val_predictions = np.argmax(val_outputs, axis=1)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Predictions: {np.unique(val_predictions)}")
        print(f"Val Labels: {np.unique(val_labels.astype(int))}")

        if val_auc > best_auc:
            best_auc = val_auc
        if val_acc > best_acc:
            best_acc = val_acc
        if val_f1 > best_f1:
            best_f1 = val_f1

        if val_loss < best_val_loss and epoch >10:
            best_val_loss = val_loss
            # Use deep copy to ensure model state is not overwritten by subsequent training, move to CPU to save GPU memory
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_auc'].append(val_auc)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(val_f1)

    # Save best model
    model_path = os.path.join(split_output_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    print(f"✓ Saved best model from epoch {best_epoch}! Val Loss: {best_val_loss:.4f} "
          f"(AUC: {best_auc:.4f}, Acc: {best_acc:.4f}, F1: {best_f1:.4f})")

    np.save(os.path.join(split_output_dir, 'training_history.npy'), training_history)

    # Load best model for testing and validation
    model.load_state_dict(best_model_state)
    print(f"Using best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # Get best model's training results
    best_train_auc, best_train_acc, best_train_f1, best_train_outputs, best_train_labels = test_model(model, train_loader, device)
    if best_train_outputs is None:
        print("Warning: Could not get best model training results")
        best_train_auc = 0.0
        best_train_acc = 0.0
        best_train_f1 = 0.0
        best_train_outputs = np.array([])
        best_train_labels = np.array([])
    
    # Get best model's validation results
    best_val_auc, best_val_acc, best_val_f1, best_val_outputs, best_val_labels = test_model(model, val_loader, device)
    if best_val_outputs is None:
        print("Warning: Could not get best model validation results")
        best_val_outputs = val_outputs
        best_val_labels = val_labels
        best_val_auc = val_auc
        best_val_acc = val_acc
        best_val_f1 = val_f1
    
    # Get best model's test results
    test_auc, test_acc, test_f1, test_outputs, test_labels = test_model(model, test_loader, device)

    if test_outputs is None:
        return None

    test_predictions = np.argmax(test_outputs, axis=1)
    best_val_predictions = np.argmax(best_val_outputs, axis=1)
    best_train_predictions = np.argmax(best_train_outputs, axis=1) if len(best_train_outputs) > 0 else np.array([])
    
    # Calculate per-class accuracy for test set
    per_class_acc = {}
    unique_labels = np.unique(test_labels)
    for label in unique_labels:
        mask = test_labels == label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(test_labels[mask], test_predictions[mask])
            per_class_acc[int(label)] = class_acc
    
    # Calculate per-class accuracy for validation set
    per_class_val_acc = {}
    unique_val_labels = np.unique(best_val_labels)
    for label in unique_val_labels:
        mask = best_val_labels == label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(best_val_labels[mask], best_val_predictions[mask])
            per_class_val_acc[int(label)] = class_acc
    
    # Calculate per-class accuracy for training set
    per_class_train_acc = {}
    if len(best_train_labels) > 0:
        unique_train_labels = np.unique(best_train_labels)
        for label in unique_train_labels:
            mask = best_train_labels == label
            if np.sum(mask) > 0:
                class_acc = accuracy_score(best_train_labels[mask], best_train_predictions[mask])
                per_class_train_acc[int(label)] = class_acc

    results = {
        'model_name': model_name,
        'split_idx': split_idx,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_auc': float(best_auc),
        'best_val_acc': float(best_acc),
        'best_val_f1': float(best_f1),
        'best_per_class_acc': per_class_acc,
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'flops': flops_str,
        'flops_value': float(flops_value),
        'parameters': params_str,
        'parameters_value': int(params_value),
        'test_outputs': test_outputs.tolist() if isinstance(test_outputs, np.ndarray) else test_outputs,
        'test_labels': test_labels.tolist() if isinstance(test_labels, np.ndarray) else test_labels,
        'test_probabilities': torch.softmax(torch.from_numpy(test_outputs), dim=1).numpy().tolist() if isinstance(test_outputs, np.ndarray) else test_outputs,
        'model_weights_path': model_path,
        # Add best model training results
        'best_train_auc': float(best_train_auc),
        'best_train_acc': float(best_train_acc),
        'best_train_f1': float(best_train_f1),
        'best_train_outputs': best_train_outputs.tolist() if isinstance(best_train_outputs, np.ndarray) else best_train_outputs,
        'best_train_labels': best_train_labels.tolist() if isinstance(best_train_labels, np.ndarray) else best_train_labels,
        'best_train_probabilities': torch.softmax(torch.from_numpy(best_train_outputs), dim=1).numpy().tolist() if isinstance(best_train_outputs, np.ndarray) else best_train_outputs,
        'best_train_predictions': best_train_predictions.tolist() if isinstance(best_train_predictions, np.ndarray) else best_train_predictions,
        'best_per_class_train_acc': per_class_train_acc,
        # Add best model validation results
        'best_val_outputs': best_val_outputs.tolist() if isinstance(best_val_outputs, np.ndarray) else best_val_outputs,
        'best_val_labels': best_val_labels.tolist() if isinstance(best_val_labels, np.ndarray) else best_val_labels,
        'best_val_probabilities': torch.softmax(torch.from_numpy(best_val_outputs), dim=1).numpy().tolist() if isinstance(best_val_outputs, np.ndarray) else best_val_outputs,
        'best_val_predictions': best_val_predictions.tolist() if isinstance(best_val_predictions, np.ndarray) else best_val_predictions,
        'best_per_class_val_acc': per_class_val_acc
    }

    # Save metrics.json
    with open(os.path.join(split_output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save test-related numpy results separately
    np.save(os.path.join(split_output_dir, 'test_outputs.npy'), test_outputs)
    np.save(os.path.join(split_output_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(split_output_dir, 'test_labels.npy'), test_labels)
    test_probabilities = np.array(results['test_probabilities'])
    np.save(os.path.join(split_output_dir, 'test_probabilities.npy'), test_probabilities)

    # Save best model training-related numpy results
    if len(best_train_outputs) > 0:
        np.save(os.path.join(split_output_dir, 'best_train_outputs.npy'), best_train_outputs)
        np.save(os.path.join(split_output_dir, 'best_train_predictions.npy'), best_train_predictions)
        np.save(os.path.join(split_output_dir, 'best_train_labels.npy'), best_train_labels)
        best_train_probabilities = np.array(results['best_train_probabilities'])
        np.save(os.path.join(split_output_dir, 'best_train_probabilities.npy'), best_train_probabilities)
    
    # Save best model validation-related numpy results
    np.save(os.path.join(split_output_dir, 'best_val_outputs.npy'), best_val_outputs)
    np.save(os.path.join(split_output_dir, 'best_val_predictions.npy'), best_val_predictions)
    np.save(os.path.join(split_output_dir, 'best_val_labels.npy'), best_val_labels)
    best_val_probabilities = np.array(results['best_val_probabilities'])
    np.save(os.path.join(split_output_dir, 'best_val_probabilities.npy'), best_val_probabilities)

    # Backward compatibility: save in unified format
    np.save(os.path.join(split_output_dir, 'test_results.npy'), {
        'test_outputs': test_outputs,
        'test_predictions': test_predictions,
        'test_labels': test_labels,
        'test_probabilities': results['test_probabilities']
    })
    
    # Save best model training results
    if len(best_train_outputs) > 0:
        np.save(os.path.join(split_output_dir, 'best_train_results.npy'), {
            'best_train_outputs': best_train_outputs,
            'best_train_predictions': best_train_predictions,
            'best_train_labels': best_train_labels,
            'best_train_probabilities': results['best_train_probabilities']
        })
    
    # Save best model validation results
    np.save(os.path.join(split_output_dir, 'best_val_results.npy'), {
        'best_val_outputs': best_val_outputs,
        'best_val_predictions': best_val_predictions,
        'best_val_labels': best_val_labels,
        'best_val_probabilities': results['best_val_probabilities']
    })

    # Feature extraction
    train_features, train_labels, train_outputs = extract_features(model, train_loader, device, "training")
    if train_features is not None:
        save_features_to_h5(train_features, train_labels, train_outputs,
                             os.path.join(split_output_dir, 'train_features_labels.h5'), "training")
    val_features, val_labels, val_outputs = extract_features(model, val_loader, device, "validation")
    if val_features is not None:
        save_features_to_h5(val_features, val_labels, val_outputs,
                             os.path.join(split_output_dir, 'val_features_labels.h5'), "validation")
    test_features, test_labels, test_outputs = extract_features(model, test_loader, device, "test")
    if test_features is not None:
        save_features_to_h5(test_features, test_labels, test_outputs,
                             os.path.join(split_output_dir, 'test_features_labels.h5'), "test")

    return results

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='ABMIL Training Script with Feature Extraction (Single GPU)')
    parser.add_argument('--dataset', type=str, default='brca', choices=['crc', 'bracs', 'bracs_7class', 'brca', 'panda'],
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
        model_name = MODEL_NAME

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
        # PANDA currently uses only one splits_0_k.csv, so run 1 split; other datasets keep 5 splits
        num_splits = 1 if DATASET_TYPE == 'panda' else 5
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
                continue

        if not all_results:
            print(f"No successful splits for model {MODEL_NAME}")
            continue

        # Summary
        avg_best_val_loss = np.mean([r['best_val_loss'] for r in all_results])
        avg_best_val_auc = np.mean([r['best_val_auc'] for r in all_results])
        avg_best_val_acc = np.mean([r['best_val_acc'] for r in all_results])
        avg_best_val_f1 = np.mean([r['best_val_f1'] for r in all_results])
        avg_test_auc = np.mean([r['test_auc'] for r in all_results])
        avg_test_acc = np.mean([r['test_acc'] for r in all_results])
        avg_test_f1 = np.mean([r['test_f1'] for r in all_results])

        std_best_val_loss = np.std([r['best_val_loss'] for r in all_results])
        std_best_val_auc = np.std([r['best_val_auc'] for r in all_results])
        std_best_val_acc = np.std([r['best_val_acc'] for r in all_results])
        std_best_val_f1 = np.std([r['best_val_f1'] for r in all_results])
        std_test_auc = np.std([r['test_auc'] for r in all_results])
        std_test_acc = np.std([r['test_acc'] for r in all_results])
        std_test_f1 = np.std([r['test_f1'] for r in all_results])

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
                'model_name': MODEL_NAME
            },
            'average_results': {
                'avg_best_val_loss': float(avg_best_val_loss),
                'avg_best_val_auc': float(avg_best_val_auc),
                'avg_best_val_acc': float(avg_best_val_acc),
                'avg_best_val_f1': float(avg_best_val_f1),
                'avg_test_auc': float(avg_test_auc),
                'avg_test_acc': float(avg_test_acc),
                'avg_test_f1': float(avg_test_f1),
                'std_best_val_loss': float(std_best_val_loss),
                'std_best_val_auc': float(std_best_val_auc),
                'std_best_val_acc': float(std_best_val_acc),
                'std_best_val_f1': float(std_best_val_f1),
                'std_test_auc': float(std_test_auc),
                'std_test_acc': float(std_test_acc),
                'std_test_f1': float(std_test_f1)
            },
            'individual_results': all_results
        }

        with open(os.path.join(output_base_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(output_base_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSummary for model {MODEL_NAME}:")
        print(f"  Avg best val AUC: {avg_best_val_auc:.4f} ± {std_best_val_auc:.4f}")
        print(f"  Avg best val Acc: {avg_best_val_acc:.4f} ± {std_best_val_acc:.4f}")
        print(f"  Avg best val F1: {avg_best_val_f1:.4f} ± {std_best_val_f1:.4f}")
        print(f"  Avg test AUC: {avg_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"  Avg test Acc: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"  Avg test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"Results saved to: {output_base_dir}")
        print(f"  - Best model training results saved as: best_train_*.npy")
        print(f"  - Best model validation results saved as: best_val_*.npy")
        print(f"  - Best model training/validation results included in: metrics.json")

if __name__ == "__main__":
    main()