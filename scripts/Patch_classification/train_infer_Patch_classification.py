#!/usr/bin/env python3
"""
Patch-level Classification Training - GitHub Release Version

This script trains patch-level classifiers using pre-extracted features and saves
logits for train/val/test sets using the best model. It supports multiple patch
classification datasets and uses MLP classifiers with LayerNorm and GELU activation.

Key characteristics:
  - Trains MLP classifiers on pre-extracted patch features.
  - Saves best model's validation and test results (outputs, predictions, labels, probabilities).
  - Saves logits for all datasets (train/val/test) with consistent order for fusion tasks.
  - Supports multiple datasets (CCRCC, CRC-100K, CRC-MSI, TCGA-TILs, BACH, ESCA, PCAM).
  - Uses pre-extracted features from extract_feature.py.

Supported datasets:
  - CCRCC: 4 categories (cancer, stroma, normal, blood)
  - CRC-100K: 9 categories (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM)
  - CRC-MSI: 2 categories (MSIH, nonMSIH) - binary classification
  - TCGA-TILs: 2 categories (til-positive, til-negative) - binary classification
  - BACH: 4 categories (Benign, InSitu, Invasive, Normal)
  - ESCA: 11 categories (ADVENT, LAM_PROP, MUSC_MUC, MUSC_PROP, REGR_TU, SH_MAG, SH_OES, SUB_GL, SUBMUC, TUMOR, ULCUS)
  - PCAM: 2 categories (normal, tumor) - binary classification

Expected directory structure:
  <features_dir>/
    - {model_name}_train_features.pt (pre-extracted train features)
    - {model_name}_test_features.pt (pre-extracted test features)
    - {model_name}_val_features.pt (pre-extracted val features, optional, for TCGA-TILs/PCAM)

Output directory structure:
  <output_dir>/
    - best_model.pth (saved model checkpoint)
    - metrics.json (training metrics)
    - best_train_outputs.npy, best_val_outputs.npy, test_outputs.npy
    - best_train_labels.npy, best_val_labels.npy, test_labels.npy
    - best_train_predictions.npy, best_val_predictions.npy, test_predictions.npy
    - best_train_probabilities.npy, best_val_probabilities.npy, test_probabilities.npy
    - training_history.npy
    - summary.json

Usage example:
  python train_patch_classifier.py --dataset ccrcc \
      --features_dir /path/to/features \
      --output_dir /path/to/outputs \
      --batch_size 16
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Add scripts_tile directory to Python path for dataset imports
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_tile_dir = os.path.abspath(os.path.join(script_dir, '../../../scripts_tile'))
if scripts_tile_dir not in sys.path:
    sys.path.insert(0, scripts_tile_dir)

# -------------------------------
# Dataset imports
# -------------------------------
try:
    from CCRCC_dataset import CCRCCDataset
    CCRCC_AVAILABLE = True
except ImportError:
    CCRCC_AVAILABLE = False
    print("Warning: CCRCC_dataset not available")

try:
    from crc_100k_dataset import CRCDataset
    CRC_AVAILABLE = True
except ImportError:
    CRC_AVAILABLE = False
    print("Warning: crc_100k_dataset not available")

try:
    from CRC_MSI_dataset import CRCMSIDataset
    CRC_MSI_AVAILABLE = True
except ImportError:
    CRC_MSI_AVAILABLE = False
    print("Warning: CRC_MSI_dataset not available")

try:
    from TCGA_TILs_dataset import TCGATILsDataset
    TCGA_TILS_AVAILABLE = True
except ImportError:
    TCGA_TILS_AVAILABLE = False
    print("Warning: TCGA_TILs_dataset not available")

try:
    from BACH_dataset import BACHDataset
    BACH_AVAILABLE = True
except ImportError:
    BACH_AVAILABLE = False
    print("Warning: BACH_dataset not available")

try:
    from ESCA_dataset import ESCADataset
    ESCA_AVAILABLE = True
except ImportError:
    ESCA_AVAILABLE = False
    print("Warning: ESCA_dataset not available")

try:
    from PCAM_dataset import PCAMDataset
    PCAM_AVAILABLE = True
except ImportError:
    PCAM_AVAILABLE = False
    print("Warning: PCAM_dataset not available")

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
# Dataset and model
# -------------------------------
class FeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted features and labels from .pt files.
    """
    def __init__(self, features_path):
        """
        Initialize feature dataset.
        
        Args:
            features_path: str - Path to .pt file containing {'features': tensor, 'labels': tensor}
        """
        data = torch.load(features_path)
        self.features = data['features']  # [N, feature_dim]
        self.labels = data['labels']      # [N]
        
    def __len__(self):
        """Return dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get item by index.
        
        Args:
            idx: int - Index of the item
        
        Returns:
            tuple: (features, label)
        """
        return self.features[idx], self.labels[idx]

class MLPFeatureClassifier(nn.Module):
    """
    MLP classifier for patch-level features with LayerNorm and GELU activation.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: int - Input feature dimension
            hidden_dim: int - Hidden dimension (default: 128)
            num_classes: int - Number of classes (default: 10)
        """
        super(MLPFeatureClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.GELU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: torch.Tensor - Input features
        
        Returns:
            torch.Tensor - Class predictions
        """
        features = self.mlp(x)
        predictions = self.classifier(features)
        return predictions

# -------------------------------
# Training and validation
# -------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: MLPFeatureClassifier instance
        train_loader: DataLoader for training
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
    
    Returns:
        tuple: (avg_loss, avg_acc)
    """
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
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
    Compute metrics from logits: convert to probabilities and calculate AUC/ACC/F1.
    
    Args:
        all_outputs: np.ndarray - Model outputs (logits)
        all_labels: np.ndarray - Ground truth labels
    
    Returns:
        tuple: (auc, accuracy, f1, probabilities, predictions)
    """
    all_labels = np.asarray(all_labels, dtype=np.int64).ravel()
    preds = np.argmax(all_outputs, axis=1)
    probabilities = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
    
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(all_labels, probabilities[:, 1])
        try:
            f1 = f1_score(all_labels, preds, zero_division=0)
        except ValueError:
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
    else:
        auc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    
    accuracy = accuracy_score(all_labels, preds)
    return auc, accuracy, f1, probabilities, preds

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: MLPFeatureClassifier instance
        val_loader: DataLoader for validation
        criterion: Loss function
        device: torch device
    
    Returns:
        tuple: (avg_loss, auc, accuracy, f1, outputs, labels)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
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
    """
    Test the model.
    
    Args:
        model: MLPFeatureClassifier instance
        test_loader: DataLoader for testing
        device: torch device
    
    Returns:
        tuple: (auc, accuracy, f1, outputs, labels)
    """
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    if len(all_outputs) == 0:
        return None, None, None, None, None
    
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc, accuracy, f1, probabilities, preds = compute_metrics_from_logits(all_outputs, all_labels)
    return auc, accuracy, f1, all_outputs, all_labels

# -------------------------------
# Main training function
# -------------------------------
def train_single_gpu(train_features_path, test_features_path, output_dir, device,
                     input_feature_dim, num_epochs, model_name, dataset_type='ccrcc', 
                     batch_size=16, hidden_dim=128, random_state=42, val_features_path=None):
    """
    Train a single model (no cross-validation splits).
    
    Args:
        train_features_path: str - Path to train features .pt file
        test_features_path: str - Path to test features .pt file (used for both val and test if val not provided)
        output_dir: str - Output directory for this model
        device: torch device - Device to use
        input_feature_dim: int - Input feature dimension
        num_epochs: int - Number of training epochs
        model_name: str - Name of the model
        dataset_type: str - Dataset type ('ccrcc', 'crc', etc.)
        batch_size: int - Batch size (default: 16)
        hidden_dim: int - Hidden dimension for MLP (default: 128)
        random_state: int - Random seed (default: 42)
        val_features_path: str or None - Path to val features .pt file (optional)
    
    Returns:
        dict or None: Training results
    """
    print(f"Training model {model_name} on device {device}")
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    # Load train features
    if not os.path.exists(train_features_path):
        print(f"Error: Train features file does not exist: {train_features_path}")
        return None
    
    # Load val/test features
    if val_features_path is not None:
        if not os.path.exists(val_features_path):
            print(f"Error: Val features file does not exist: {val_features_path}")
            return None
    if not os.path.exists(test_features_path):
        print(f"Error: Test features file does not exist: {test_features_path}")
        return None
    
    # Load train features and labels
    train_dataset = FeatureDataset(train_features_path)
    train_features = train_dataset.features.numpy()
    train_labels = train_dataset.labels.numpy()
    
    # Load val/test features and labels
    if val_features_path is not None:
        val_dataset = FeatureDataset(val_features_path)
        val_features = val_dataset.features.numpy()
        val_labels = val_dataset.labels.numpy()
    else:
        val_features = None
        val_labels = None

    test_dataset = FeatureDataset(test_features_path)
    test_features = test_dataset.features.numpy()
    test_labels = test_dataset.labels.numpy()
    
    # If val not provided, reuse test as val (old behavior)
    if val_features is None:
        val_features = test_features
        val_labels = test_labels
        print(f"Note: val and test share the same dataset (test_features) because val_features_path not provided")
    else:
        print(f"Using distinct val and test datasets")
    
    print(f"Data split: train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")
    
    # Create datasets for training (train needs shuffle, val/test don't)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels).long()
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_labels).long()
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features).float(),
        torch.from_numpy(test_labels).long()
    )
    
    # Create data loaders for training (train shuffled, val/test not shuffled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Create ordered datasets for saving logits (no shuffle, ensure consistent order)
    train_dataset_ordered = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels).long()
    )
    val_dataset_ordered = torch.utils.data.TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_labels).long()
    )
    test_dataset_ordered = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features).float(),
        torch.from_numpy(test_labels).long()
    )
    
    # Create ordered data loaders for saving logits (no shuffle, consistent order)
    train_loader_ordered = DataLoader(train_dataset_ordered, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_loader_ordered = DataLoader(val_dataset_ordered, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader_ordered = DataLoader(test_dataset_ordered, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Determine number of classes
    if dataset_type == 'ccrcc':
        n_classes = 4  # cancer, stroma, normal, blood (only four categories)
    elif dataset_type == 'crc':
        n_classes = 9  # ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
    elif dataset_type == 'bach':
        n_classes = 4  # Benign, InSitu, Invasive, Normal
    elif dataset_type == 'esca':
        n_classes = 11  # ADVENT, LAM_PROP, MUSC_MUC, MUSC_PROP, REGR_TU, SH_MAG, SH_OES, SUB_GL, SUBMUC, TUMOR, ULCUS
    elif dataset_type == 'pcam':
        n_classes = 2  # normal, tumor (binary classification)
    else:
        # Get unique classes from both train and test labels
        all_labels_combined = np.concatenate([train_labels, test_labels])
        n_classes = len(np.unique(all_labels_combined))
    
    # Model
    model = MLPFeatureClassifier(
        input_dim=input_feature_dim,
        hidden_dim=hidden_dim,
        num_classes=n_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    best_model_auc = 0.0  # Metrics from the best model (based on val_loss)
    best_model_acc = 0.0
    best_model_f1 = 0.0
    last_val_outputs = None
    last_val_labels = None
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
        
        # Store last validation outputs for fallback
        last_val_outputs = val_outputs
        last_val_labels = val_labels
        
        val_predictions = np.argmax(val_outputs, axis=1)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Track best metrics for reporting (but don't use for model selection)
        if val_auc > best_auc:
            best_auc = val_auc
        if val_acc > best_acc:
            best_acc = val_acc
        if val_f1 > best_f1:
            best_f1 = val_f1

        # Save best model based on validation loss (after warmup period)
        # Note: Model selection uses val_loss (more stable), not AUC/Acc/F1
        # The tracked best_auc/acc/f1 are for reporting purposes only
        if val_loss < best_val_loss and epoch >= 10:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            # Save metrics from the best model (based on val_loss)
            best_model_auc = val_auc
            best_model_acc = val_acc
            best_model_f1 = val_f1

        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_auc'].append(val_auc)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(val_f1)

    # Check if we have a saved model
    if best_model_state is None:
        print("Warning: No model was saved during training (all epochs had val_loss >= best_val_loss or epoch < 10)")
        print("Using the last epoch's model state instead.")
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = num_epochs
        if last_val_outputs is not None:
            # Use last validation metrics
            best_val_loss = float('inf')  # We don't have the actual loss, but this is just for logging
            val_auc, val_acc, val_f1, _, _ = compute_metrics_from_logits(last_val_outputs, last_val_labels)
            best_auc = val_auc
            best_acc = val_acc
            best_f1 = val_f1
            best_model_auc = val_auc  # Update best_model_* metrics
            best_model_acc = val_acc
            best_model_f1 = val_f1
    
    # Save best model
    model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    # Use best_model_* metrics (from the model selected by val_loss) for reporting
    print(f"✓ Saved best model from epoch {best_epoch}! Val Loss: {best_val_loss:.4f} "
          f"(AUC: {best_model_auc:.4f}, Acc: {best_model_acc:.4f}, F1: {best_model_f1:.4f})")

    np.save(os.path.join(output_dir, 'training_history.npy'), training_history)

    # Load best model for testing
    model.load_state_dict(best_model_state)
    print(f"Using best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # Re-infer with ordered loaders to ensure consistent order for logits saving
    # IMPORTANT: This ensures all models have logits in the same order for fusion
    # - Training uses shuffled train_loader (for better training)
    # - Logits saving uses ordered loaders (no shuffle) to ensure consistent order
    # - All models will have logits aligned by sample index for fusion training
    print("Re-inferring with ordered data loaders (no shuffle) to save logits...")
    print("  This ensures consistent order across all models for fusion training")
    
    # Get best model's training results (ordered, no shuffle)
    best_train_auc, best_train_acc, best_train_f1, best_train_outputs, best_train_labels = test_model(model, train_loader_ordered, device)
    if best_train_outputs is None:
        print("Warning: Could not get best model training results")
        best_train_auc = 0.0
        best_train_acc = 0.0
        best_train_f1 = 0.0
        best_train_outputs = np.array([])
        best_train_labels = np.array([])
    
    # Get best model's validation results (ordered, no shuffle)
    val_auc, val_acc, val_f1, val_outputs, val_labels = test_model(model, val_loader_ordered, device)
    if val_outputs is None:
        print("Warning: Could not get best model val results")
        # Fallback to last validation outputs if available
        if last_val_outputs is not None:
            val_outputs = last_val_outputs
            val_labels = last_val_labels
            val_auc, val_acc, val_f1, _, _ = compute_metrics_from_logits(val_outputs, val_labels)
        else:
            print("Error: No val outputs available. Cannot proceed.")
            return None
    
    # Get best model's test results (ordered, no shuffle)
    test_auc, test_acc, test_f1, test_outputs, test_labels = test_model(model, test_loader_ordered, device)
    if test_outputs is None:
        print("Warning: Could not get best model test results")
        # Fallback to val outputs if available
        test_outputs = val_outputs
        test_labels = val_labels
        test_auc, test_acc, test_f1, _, _ = compute_metrics_from_logits(test_outputs, test_labels)
    
    # Update best_model_* metrics with actual metrics from best model inference
    best_model_auc = val_auc
    best_model_acc = val_acc
    best_model_f1 = val_f1
    best_val_outputs = val_outputs
    best_val_labels = val_labels

    test_predictions = np.argmax(test_outputs, axis=1)
    best_val_predictions = np.argmax(best_val_outputs, axis=1)
    best_train_predictions = np.argmax(best_train_outputs, axis=1) if len(best_train_outputs) > 0 else np.array([])
    
    results = {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_auc': float(best_model_auc),  # Use metrics from best model (based on val_loss)
        'best_val_acc': float(best_model_acc),
        'best_val_f1': float(best_model_f1),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'test_outputs': test_outputs.tolist() if isinstance(test_outputs, np.ndarray) else test_outputs,
        'test_labels': test_labels.tolist() if isinstance(test_labels, np.ndarray) else test_labels,
        'test_probabilities': torch.softmax(torch.from_numpy(test_outputs), dim=1).numpy().tolist() if isinstance(test_outputs, np.ndarray) else test_outputs,
        'model_weights_path': model_path,
        'best_train_auc': float(best_train_auc),
        'best_train_acc': float(best_train_acc),
        'best_train_f1': float(best_train_f1),
        'best_train_outputs': best_train_outputs.tolist() if isinstance(best_train_outputs, np.ndarray) else best_train_outputs,
        'best_train_labels': best_train_labels.tolist() if isinstance(best_train_labels, np.ndarray) else best_train_labels,
        'best_train_probabilities': torch.softmax(torch.from_numpy(best_train_outputs), dim=1).numpy().tolist() if isinstance(best_train_outputs, np.ndarray) else best_train_outputs,
        'best_train_predictions': best_train_predictions.tolist() if isinstance(best_train_predictions, np.ndarray) else best_train_predictions,
        'best_val_outputs': best_val_outputs.tolist() if isinstance(best_val_outputs, np.ndarray) else best_val_outputs,
        'best_val_labels': best_val_labels.tolist() if isinstance(best_val_labels, np.ndarray) else best_val_labels,
        'best_val_probabilities': torch.softmax(torch.from_numpy(best_val_outputs), dim=1).numpy().tolist() if isinstance(best_val_outputs, np.ndarray) else best_val_outputs,
        'best_val_predictions': best_val_predictions.tolist() if isinstance(best_val_predictions, np.ndarray) else best_val_predictions
    }

    # Save metrics.json
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save numpy arrays
    np.save(os.path.join(output_dir, 'test_outputs.npy'), test_outputs)
    np.save(os.path.join(output_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    test_probabilities = np.array(results['test_probabilities'])
    np.save(os.path.join(output_dir, 'test_probabilities.npy'), test_probabilities)

    if len(best_train_outputs) > 0:
        np.save(os.path.join(output_dir, 'best_train_outputs.npy'), best_train_outputs)
        np.save(os.path.join(output_dir, 'best_train_predictions.npy'), best_train_predictions)
        np.save(os.path.join(output_dir, 'best_train_labels.npy'), best_train_labels)
        best_train_probabilities = np.array(results['best_train_probabilities'])
        np.save(os.path.join(output_dir, 'best_train_probabilities.npy'), best_train_probabilities)
    
    np.save(os.path.join(output_dir, 'best_val_outputs.npy'), best_val_outputs)
    np.save(os.path.join(output_dir, 'best_val_predictions.npy'), best_val_predictions)
    np.save(os.path.join(output_dir, 'best_val_labels.npy'), best_val_labels)
    best_val_probabilities = np.array(results['best_val_probabilities'])
    np.save(os.path.join(output_dir, 'best_val_probabilities.npy'), best_val_probabilities)

    return results

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Patch-level Classification Training Script')
    parser.add_argument('--dataset', type=str, default='bach', 
                        choices=['ccrcc', 'crc', 'crc_msi', 'tcga_tils', 'bach', 'esca', 'pcam'],
                        help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing pre-extracted features (.pt files). Supports {MODEL_NAME} placeholder.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving results. Supports {MODEL_NAME} placeholder.')
    args = parser.parse_args()
    
    DATASET_TYPE = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (single GPU)")
    
    # Model hyperparameters (from meta.json or default)
    MODEL_HYPERPARAMS = {
        "conch_v15":    {"dim": 768,  "epochs": 30},
        "uni_v2":       {"dim": 1536, "epochs": 30},
        "phikon_v2":    {"dim": 1024, "epochs": 30},
        "virchow2":     {"dim": 2560, "epochs": 30},
        "gigapath":     {"dim": 1536, "epochs": 30},
        "hoptimus1":    {"dim": 1536, "epochs": 30},
        "kaiko-vitl14": {"dim": 1024, "epochs": 30},
        "lunit-vits8":  {"dim": 384,  "epochs": 30},
        "ctranspath":   {"dim": 768,  "epochs": 30}
    }
    
    for MODEL_NAME, model_params in MODEL_HYPERPARAMS.items():
        INPUT_FEATURE_DIM = model_params["dim"]
        NUM_EPOCHS = model_params["epochs"]

        # Support {MODEL_NAME} placeholder in features_dir and output_dir
        if '{MODEL_NAME}' in args.features_dir:
            features_dir = args.features_dir.format(MODEL_NAME=MODEL_NAME)
        else:
            features_dir = args.features_dir

        if '{MODEL_NAME}' in args.output_dir:
            output_dir = args.output_dir.format(MODEL_NAME=MODEL_NAME)
        else:
            output_dir = os.path.join(args.output_dir, MODEL_NAME)

        # Features file paths (train/val/test are already separated during feature extraction)
        train_features_path = os.path.join(features_dir, f"{MODEL_NAME}_train_features.pt")
        test_features_path = os.path.join(features_dir, f"{MODEL_NAME}_test_features.pt")
        val_features_path = os.path.join(features_dir, f"{MODEL_NAME}_val_features.pt")
        
        # Check if val features file exists, if not set to None
        if not os.path.exists(val_features_path):
            val_features_path = None
        
        print(f"\n{'='*80}")
        print(f"Training model: {MODEL_NAME} on dataset: {DATASET_TYPE}")
        print(f"Train features path: {train_features_path}")
        print(f"Test features path: {test_features_path}")
        if val_features_path is not None:
            print(f"Val features path: {val_features_path}")
        print(f"Input dim: {INPUT_FEATURE_DIM}, epochs: {NUM_EPOCHS}, batch_size: {args.batch_size}")
        print(f"{'='*80}")

        if not os.path.exists(train_features_path):
            print(f"Error: train features file does not exist: {train_features_path}")
            continue
        
        if not os.path.exists(test_features_path):
            print(f"Error: test features file does not exist: {test_features_path}")
            continue
        if val_features_path is not None and not os.path.exists(val_features_path):
            print(f"Error: val features file does not exist: {val_features_path}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        
        try:
            result = train_single_gpu(
                train_features_path, test_features_path, output_dir,
                device, INPUT_FEATURE_DIM, NUM_EPOCHS, MODEL_NAME,
                DATASET_TYPE, args.batch_size, 128, SEED,
                val_features_path=val_features_path
            )
            
            if result is None:
                print(f"Training failed for model {MODEL_NAME}")
                continue

            total_time = time.time() - start_time

            summary = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'total_time_seconds': total_time,
                'model_name': MODEL_NAME,
                'config': {
                    'input_feature_dim': INPUT_FEATURE_DIM,
                    'hidden_dim': 128,
                    'num_epochs': NUM_EPOCHS,
                    'batch_size': args.batch_size,
                    'train_features_path': train_features_path,
                    'test_features_path': test_features_path,
                    'val_features_path': val_features_path,
                    'model_name': MODEL_NAME
                },
                'results': result
            }

            with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            print(f"\nResults for model {MODEL_NAME}:")
            print(f"  Best val AUC: {result['best_val_auc']:.4f}")
            print(f"  Best val Acc: {result['best_val_acc']:.4f}")
            print(f"  Best val F1: {result['best_val_f1']:.4f}")
            print(f"  Test AUC: {result['test_auc']:.4f}")
            print(f"  Test Acc: {result['test_acc']:.4f}")
            print(f"  Test F1: {result['test_f1']:.4f}")
            print(f"Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"Training failed for model {MODEL_NAME}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
