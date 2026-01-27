#!/usr/bin/env python3
"""
POE Logit-based Fusion for Patch-level Classification - GitHub Release Version

This script trains a lightweight gating network to fuse multiple expert models'
logits on patch-level classification tasks.

Key characteristics:
  - Only uses experts' logits to build gating features (no patch embeddings used).
  - Uses training set predictions to estimate an expert-correlation matrix R for regularization.
  - Produces validation/test metrics and summary JSON outputs.
  - Patch-level classification has no cross-validation splits (single training run).

Expected directory structure:
  <root>/<model_name>/
    - best_train_outputs.npy (train logits)
    - best_train_labels.npy (train labels)
    - best_val_outputs.npy (val logits)
    - best_val_labels.npy (val labels)
    - test_outputs.npy (test logits)
    - test_labels.npy (test labels)

Output directory structure:
  <root>/logits_fusion_results/
    - poe_val_summary.json (validation metrics)
    - poe_test_summary.json (test metrics)
    - poe_val_per_split.json (per-split format, split=0)
    - poe_test_per_split.json (per-split format, split=0)
    - test_outputs.npy, test_predictions.npy, test_labels.npy, test_probabilities.npy
    - val_outputs.npy, val_predictions.npy, val_labels.npy, val_probabilities.npy

Usage example:
  python logitsfed_Patch_classification.py --root /path/to/patch_results \
      --models conch_v15 uni_v2 phikon_v2
"""

import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Model: POE (logit-based gating)
# -------------------------------
class POE(nn.Module):
    """
    Product-of-Experts style fusion with a learnable gating network.

    Inputs:
      x      : [B, D] gating feature vector extracted from experts' logits
      logits : [B, M, K] experts' logits

    Outputs:
      fused_logits : [B, K] fused logits (compatible with CrossEntropyLoss)
      weights      : [B, M] sample-wise expert weights (Softmax normalized)
    """
    def __init__(
        self,
        in_dim: int,
        n_models: int,
        hidden: int = 0,
        gate_temp: float = 1.0,
        init_uniform: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.n_models = n_models
        self.gate_temp = gate_temp

        if hidden and hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, n_models, bias=bias),
            )
        else:
            self.gate = nn.Linear(in_dim, n_models, bias=bias)

        # Initialize last layer to make weights nearly uniform at start
        if init_uniform:
            if isinstance(self.gate, nn.Linear):
                nn.init.zeros_(self.gate.weight)
                if bias:
                    nn.init.zeros_(self.gate.bias)
            else:
                nn.init.zeros_(self.gate[-1].weight)
                if bias:
                    nn.init.zeros_(self.gate[-1].bias)
        else:
            self.apply(self._init_xavier)

    @staticmethod
    def _init_xavier(m):
        """Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, logits: torch.Tensor):
        """
        Forward pass.

        Args:
            x: [B, D] gating features
            logits: [B, M, K] expert logits

        Returns:
            fused_logits: [B, K] fused log-probabilities
            weights: [B, M] expert weights
        """
        raw = self.gate(x) / max(self.gate_temp, 1e-8)   # [B, M]
        weights = F.softmax(raw, dim=-1)                 # [B, M]
        log_p = F.log_softmax(logits, dim=-1)            # [B, M, K]
        fused_logits = torch.einsum('bm,bmk->bk', weights, log_p)  # [B, K] (log-prob like)
        return fused_logits, weights

# -------------------------------
# Gating feature extraction (from logits)
# -------------------------------
@torch.no_grad()
def logit_feat_extraction(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Build gating feature vector x from experts' logits only.

    Returns:
      x: [B, 3*M + 2]
        - per-expert: MSP (max prob), margin(top1-top2), entropy
        - global: avg_entropy, mutual_info
    """
    B, M, K = logits.shape
    p = F.softmax(logits, dim=-1)
    logp = (p + eps).log()

    # Confidence features
    top2 = torch.topk(p, k=min(2, K), dim=-1).values
    msp = top2[..., 0]  # [B, M]
    margin = top2[..., 0] - top2[..., 1] if K >= 2 else torch.zeros_like(msp)

    # Uncertainty features
    ent = -(p * logp).sum(dim=-1)  # [B, M]

    avg_entropy = ent.mean(dim=1, keepdim=True)  # [B, 1]

    # Mutual information approximation
    p_bar = p.mean(dim=1)  # [B, K]
    H_bar = -(p_bar * (p_bar + eps).log()).sum(dim=-1, keepdim=True)  # [B, 1]
    mutual_info = H_bar - avg_entropy  # [B, 1]

    x = torch.cat([msp, margin, ent], dim=-1)          # [B, 3M]
    x = torch.cat([x, avg_entropy, mutual_info], dim=-1)  # [B, 3M+2]
    return x

# -------------------------------
# Expert correlation matrix R + penalty
# -------------------------------
@torch.no_grad()
def compute_R_from_oof(
    oof_logits: torch.Tensor,
    oof_labels: torch.Tensor,
    kind: str = "nll",
    shrink: float = 0.1,
    pos_only: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate expert correlation matrix R from OOF logits/labels.

    Args:
        oof_logits: [N, M, C] logits from multiple models
        oof_labels: [N] ground truth labels
        kind: Error metric type ('nll', 'brier', 'err01')
        shrink: Shrinkage factor for regularization
        pos_only: Whether to clamp negative values to 0
        eps: Numerical stability epsilon

    Returns:
        R: [M, M] correlation matrix (symmetric, PSD, diagonal=0)
    """
    oof_logits = oof_logits.to(dtype=torch.float32)
    device = oof_logits.device
    N, M, C = oof_logits.shape

    if kind == "nll":
        p = F.softmax(oof_logits, dim=-1)                  # [N,M,C]
        y = oof_labels.to(torch.long).view(-1, 1, 1).expand(-1, M, 1)     # [N,M,1]
        nll = -torch.log(p.gather(-1, y).clamp_min(eps)).squeeze(-1)  # [N,M]
        X = nll
    elif kind == "brier":
        p = F.softmax(oof_logits, dim=-1)
        onehot = torch.zeros_like(p).scatter_(-1, oof_labels.view(-1,1,1), 1.0)
        brier = (p - onehot).pow(2).sum(dim=-1)            # [N,M]
        X = brier
    elif kind == "err01":
        pred = oof_logits.argmax(dim=-1)                   # [N,M]
        X = (pred != oof_labels.view(-1,1)).float()        # [N,M] 0/1 error indicator
    else:
        raise ValueError("kind must be 'nll'|'brier'|'err01'")

    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd                                      # [N,M]
    R = (Z.t() @ Z) / (Z.shape[0] - 1)                     # [M,M]
    R = (R + R.t()) * 0.5                                  # Symmetrize

    if shrink > 0:
        diag_mean = torch.diag(R).mean()
        I = torch.eye(M, device=device)
        R = (1 - shrink) * R + shrink * diag_mean * I

    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(0.0)
    R_psd = (eigvecs * eigvals.clamp_min(0.0)) @ eigvecs.t()

    if pos_only:
        R_psd = torch.clamp(R_psd, min=0.0)
    R_psd.fill_diagonal_(0.0)

    return R_psd

def compute_penalty(
    weights: torch.Tensor,
    logits: torch.Tensor = None,
    mode: str = "A",
    R: torch.Tensor = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute penalty term for regularization.

    Two modes:
      - A: Global correlation penalty w^T R w (R from OOF error correlation/covariance, PSD/symmetric)
      - B: Sample-adaptive similarity penalty sum_{m≠n} w_m w_n * sim_mn(x),
           sim_mn based on current sample's expert probability similarity (1 - JS/ln2),
           encourages weights away from "very similar" experts

    Args:
        weights: [B, M] softmax-normalized expert weights
        logits: [B, M, K] required only for mode='B' (to estimate sample-wise expert similarity)
        mode: 'A' or 'B'
        R: [M, M] required for mode='A'; should be symmetrized and non-negative truncated
        eps: Numerical stability epsilon

    Returns:
        scalar penalty (torch.Tensor)
    """
    assert weights.dim() == 2, "weights shape must be [B, M]"
    B, M = weights.shape

    if mode.upper() == "A":
        assert R is not None and R.shape == (M, M), "R must be [M,M] for mode A"
        R_use = 0.5 * (R + R.t())
        R_use = torch.clamp(R_use, min=0.0)
        quad = torch.einsum("bm,mn,bn->b", weights, R_use, weights)   # [B]
        return quad.mean()
    elif mode.upper() == "B":
        assert logits is not None and logits.dim() == 3, "logits [B,M,K] required for mode B"
        _, M2, K = logits.shape
        assert M2 == M, "weights and logits disagree on M"
        p = F.softmax(logits, dim=-1)                                 # [B, M, K]
        logp = (p + eps).log()
        p1 = p.unsqueeze(2)
        p2 = p.unsqueeze(1)
        m  = 0.5 * (p1 + p2)                                          # [B, M, M, K]
        KL12 = (p1 * (logp.unsqueeze(2) - (m + eps).log())).sum(dim=-1)  # [B, M, M]
        KL21 = (p2 * (logp.unsqueeze(1) - (m + eps).log())).sum(dim=-1)  # [B, M, M]
        JS   = 0.5 * (KL12 + KL21)                                    # [B, M, M]
        sim = 1.0 - (JS / math.log(2.0))
        sim = sim.clamp(min=0.0, max=1.0)
        sim = sim - torch.diag_embed(sim.diagonal(dim1=1, dim2=2))
        Wouter = weights.unsqueeze(2) * weights.unsqueeze(1)          # [B, M, M]
        denom = max(M * (M - 1), 1)
        penalty = (Wouter * sim).sum(dim=(1, 2)) / denom              # [B]
        return penalty.mean()
    else:
        raise ValueError("mode must be 'A' or 'B'")

# -------------------------------
# Dataset
# -------------------------------
class LogitsDataset(Dataset):
    """
    Dataset that loads logits and labels from saved numpy files.
    """
    def __init__(self, logits: np.ndarray, labels: np.ndarray):
        """
        Initialize logits dataset.

        Args:
            logits: numpy array of shape [N, M, C] where N=samples, M=models, C=classes
            labels: numpy array of shape [N]
        """
        self.logits = torch.from_numpy(logits).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        """Return dataset size."""
        return len(self.logits)
    
    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: int - Index of the item

        Returns:
            tuple: (logits, label)
        """
        return self.logits[idx], self.labels[idx]

# -------------------------------
# Data loading
# -------------------------------
def load_logits_from_models(model_root_paths, same_val_test: bool = True):
    """
    Load logits from all models (no splits, direct loading).

    Args:
        model_root_paths: List of paths to model directories
        same_val_test: bool - Whether val and test use the same dataset

    Returns:
        tuple: (train_logits, train_labels, val_logits, val_labels, test_logits, test_labels, model_names, n_models, n_classes)
    """
    per_model_logits = []
    model_names = []
    
    for model_path in model_root_paths:
        model_name = os.path.basename(model_path)
        
        try:
            # Load logits and labels directly from model directory (no split subdirectory)
            train_logits = np.load(os.path.join(model_path, 'best_train_outputs.npy'))
            train_labels = np.load(os.path.join(model_path, 'best_train_labels.npy'))
            val_logits = np.load(os.path.join(model_path, 'best_val_outputs.npy'))
            val_labels = np.load(os.path.join(model_path, 'best_val_labels.npy'))
            test_logits = np.load(os.path.join(model_path, 'test_outputs.npy'))
            test_labels = np.load(os.path.join(model_path, 'test_labels.npy'))
            
            per_model_logits.append({
                'train_logits': train_logits,
                'train_labels': train_labels,
                'val_logits': val_logits,
                'val_labels': val_labels,
                'test_logits': test_logits,
                'test_labels': test_labels
            })
            model_names.append(model_name)
        except Exception as e:
            print(f"  ✗ Failed to load {model_path}: {e}")
            continue
    
    if not per_model_logits:
        raise RuntimeError('No model logits loaded successfully')
    
    # Stack logits: [N, M, C]
    train_logits_list = [m['train_logits'] for m in per_model_logits]
    val_logits_list = [m['val_logits'] for m in per_model_logits]
    test_logits_list = [m['test_logits'] for m in per_model_logits]
    
    # Check if all have same number of samples
    train_n_samples = [len(l) for l in train_logits_list]
    val_n_samples = [len(l) for l in val_logits_list]
    test_n_samples = [len(l) for l in test_logits_list]
    
    if len(set(train_n_samples)) > 1:
        print(f"Warning: Train samples mismatch: {train_n_samples}, using minimum")
        min_train = min(train_n_samples)
        train_logits_list = [l[:min_train] for l in train_logits_list]
        train_labels = per_model_logits[0]['train_labels'][:min_train]
    else:
        train_labels = per_model_logits[0]['train_labels']
    
    if len(set(val_n_samples)) > 1:
        print(f"Warning: Val samples mismatch: {val_n_samples}, using minimum")
        min_val = min(val_n_samples)
        val_logits_list = [l[:min_val] for l in val_logits_list]
        val_labels = per_model_logits[0]['val_labels'][:min_val]
    else:
        val_labels = per_model_logits[0]['val_labels']
    
    if len(set(test_n_samples)) > 1:
        print(f"Warning: Test samples mismatch: {test_n_samples}, using minimum")
        min_test = min(test_n_samples)
        test_logits_list = [l[:min_test] for l in test_logits_list]
        test_labels = per_model_logits[0]['test_labels'][:min_test]
    else:
        test_labels = per_model_logits[0]['test_labels']
    
    # Stack along model dimension: [N, M, C]
    train_logits = np.stack(train_logits_list, axis=1)
    val_logits = np.stack(val_logits_list, axis=1)
    test_logits = np.stack(test_logits_list, axis=1)
    
    # For some datasets (CCRCC / CRC / CRC-MSI / BACH / ESCA), there is no separate val set.
    # best_val_outputs.npy / best_val_labels.npy and test_outputs.npy / test_labels.npy
    # essentially come from the same dataset. To ensure consistency, reuse test results as val.
    if same_val_test:
        if not np.array_equal(val_labels, test_labels):
            print(f"Warning: val_labels and test_labels are not identical, but they should be the same dataset")
        if val_logits.shape != test_logits.shape:
            print(f"Warning: val_logits shape {val_logits.shape} != test_logits shape {test_logits.shape}")
            print(f"  Note: val and test use the same dataset for this preset, reusing test_logits/labels for val")
            val_logits = test_logits
            val_labels = test_labels
    
    # Get number of classes
    n_classes = train_logits.shape[2]
    n_models = len(model_names)
    
    return train_logits, train_labels, val_logits, val_labels, test_logits, test_labels, model_names, n_models, n_classes

# -------------------------------
# Training
# -------------------------------
def train_with_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lambda_l: float = 1e-1,
    R=None,
    lr=5e-4,
):
    """
    Train fusion model with validation.

    Training stage: optimize total_loss = CE + lambda_l * penalty
    Validation stage: use only CE as early stopping metric, but also print CE and penalty for observation

    Args:
        model: POE model instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: torch device
        num_epochs: Number of training epochs
        lambda_l: Penalty weight
        R: Correlation matrix for penalty
        lr: Learning rate

    Returns:
        model: Trained model
    """
    weight_decay = 1e-4

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    ce = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    patience, counter = 10, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_tot, train_ce_accum, train_pen_accum = 0.0, 0.0, 0.0

        for logits, labels in train_loader:
            logits = logits.to(device)     # [B, M, K]
            labels = labels.to(device)     # [B]
            x = logit_feat_extraction(logits)

            fused_logits, weights = model(x, logits)

            # Penalty (if compute_penalty needs logits/R etc., pass them here)
            pen = compute_penalty(weights, logits=logits, R=R, mode='A')

            loss_ce = ce(fused_logits, labels)
            loss = loss_ce + lambda_l * pen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_tot += loss.item()
            train_ce_accum += loss_ce.item()
            train_pen_accum += float(pen.detach().cpu())

        scheduler.step(epoch)

        n_tr = max(1, len(train_loader))
        avg_train_tot  = train_tot / n_tr
        avg_train_ce   = train_ce_accum / n_tr
        avg_train_pen  = train_pen_accum / n_tr

        # Validation
        model.eval()
        val_ce_accum, val_pen_accum = 0.0, 0.0
        with torch.no_grad():
            for logits, labels in val_loader:
                logits = logits.to(device)
                labels = labels.to(device)
                x = logit_feat_extraction(logits)

                fused_logits, weights = model(x, logits)

                loss_ce = ce(fused_logits, labels)
                # Penalty for observation only (not used for early stopping/model selection)
                pen = compute_penalty(weights, logits=logits, R=R, mode='A')

                val_ce_accum  += loss_ce.item()
                val_pen_accum += float(pen.detach().cpu())

        n_val = max(1, len(val_loader))
        avg_val_ce  = val_ce_accum / n_val
        avg_val_pen = val_pen_accum / n_val

        # Keep original logic: validation early stopping only looks at CE
        avg_val = avg_val_ce

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        tqdm.write(
            "[epoch {:02d}] "
            "train: total={:.4f} (ce={:.4f}, pen={:.4f}) | "
            "val:   ce={:.4f}, pen={:.4f} | best_ce={:.4f}".format(
                epoch, avg_train_tot, avg_train_ce, avg_train_pen,
                avg_val_ce, avg_val_pen, best_val_loss
            )
        )

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def eval_on_split(model: nn.Module, logits: np.ndarray, labels: np.ndarray, device: torch.device):
    """
    Evaluate fusion model on a split.

    Args:
        model: POE model instance
        logits: [N, M, C] expert logits
        labels: [N] ground truth labels
        device: torch device

    Returns:
        dict: Results containing fused_logits, fused_probs, preds, auc, acc, f1, labels
    """
    model.eval()
    B = logits.shape[0]
    batch = 64
    fused_logits_list = []
    fused_probs_list = []
    for i in range(0, B, batch):
        lm = torch.from_numpy(logits[i:i+batch]).float().to(device)
        x = logit_feat_extraction(lm)
        fused_logits, w = model(x, lm)  # POE returns log-probabilities (named as fused_logits for consistency with WSI)
        # Convert to probabilities (same as WSI: apply softmax to fused_logits)
        fused_probs = torch.softmax(fused_logits, dim=1).cpu().numpy()  # [B, K]
        fused_logits_list.append(fused_logits.cpu().numpy())
        fused_probs_list.append(fused_probs)

    fused_logits = np.concatenate(fused_logits_list, axis=0)
    fused_probs = np.concatenate(fused_probs_list, axis=0)
    preds = np.argmax(fused_logits, axis=1)
    results = {
        'fused_logits': fused_logits,
        'fused_probs': fused_probs,
        'preds': preds
    }
    if labels is not None and len(labels) == fused_probs.shape[0]:
        try:
            if fused_probs.shape[1] == 2:
                auc = roc_auc_score(labels, fused_probs[:, 1])
            else:
                auc = roc_auc_score(labels, fused_probs, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.5
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds) if fused_probs.shape[1] == 2 else f1_score(labels, preds, average='macro')
        results.update({'auc': float(auc), 'acc': float(acc), 'f1': float(f1), 'labels': labels})
    return results

def eval_single_model(logits: np.ndarray, labels: np.ndarray):
    """
    Evaluate a single model's logits (no fusion).

    Args:
        logits: [N, C] logits where N is number of samples, C is number of classes
        labels: [N] ground truth labels

    Returns:
        dict: Results containing auc, acc, f1
    """
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    results = {}
    if labels is not None and len(labels) == probs.shape[0]:
        try:
            if probs.shape[1] == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.5
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds) if probs.shape[1] == 2 else f1_score(labels, preds, average='macro')
        results.update({'auc': float(auc), 'acc': float(acc), 'f1': float(f1)})
    return results

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Patch-level Logits Fusion Training')
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Root directory containing per-model subfolders with logits.'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['phikon_v2', 'virchow2', 'gigapath', 'hoptimus1', 'kaiko-vitl14', 'lunit-vits8'],
        help='List of model folder names under <root>.'
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument(
        '--dataset',
        type=str,
        default='ccrcc',
        choices=['ccrcc', 'crc', 'crc_msi', 'tcga_tils', 'bach', 'esca', 'pcam'],
        help='Dataset type (used to determine if val and test use same dataset).'
    )
    args = parser.parse_args()

    # Determine if val and test use the same dataset
    # For CCRCC / CRC / CRC-MSI / BACH / ESCA, val and test use the same dataset (no explicit val split)
    # For TCGA-TILs / PCAM, train/val/test are independent splits
    if args.dataset is not None:
        same_val_test = args.dataset in ['ccrcc', 'crc', 'crc_msi', 'bach', 'esca']
    else:
        # Default: assume same val/test if dataset not specified
        same_val_test = True

    model_root_paths = [os.path.join(args.root, d) for d in args.models if os.path.exists(os.path.join(args.root, d))]
    if not model_root_paths:
        raise RuntimeError('No model directories found')
    
    print(f"Found {len(model_root_paths)} model directories: {[os.path.basename(p) for p in model_root_paths]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        train_logits, train_labels, val_logits, val_labels, test_logits, test_labels, model_names, n_models, n_classes = load_logits_from_models(
            model_root_paths,
            same_val_test=same_val_test,
        )

        print(f"\nLoaded logits:")
        print(f"  Train: {train_logits.shape}, Val: {val_logits.shape}, Test: {test_logits.shape}")
        print(f"  Models: {model_names}")
        print(f"  Classes: {n_classes}")

        # Use train set for training fusion model (no OOF splits at patch level)
        oof_logits = train_logits
        oof_labels = train_labels

        # Compute correlation matrix R from train set
        R = compute_R_from_oof(
            torch.from_numpy(oof_logits).to(device),
            torch.from_numpy(oof_labels).long().to(device),
            kind="nll", shrink=0.1
        )

        # Create datasets
        train_ds = LogitsDataset(train_logits, train_labels)
        val_ds = LogitsDataset(val_logits, val_labels)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=False)

        # Create model
        model = POE(
            n_models=n_models,
            in_dim=n_models*3+2,
            hidden=0,
        ).to(device)
        
        # Train
        lambda_l = 1e-1  # Penalty weight
        model = train_with_val(
            model, train_loader, val_loader, device,
            num_epochs=args.num_epochs, lambda_l=lambda_l, lr=args.lr, R=R
        )

        # Evaluate fusion model on VAL and TEST
        print("\nEvaluating fusion model...")
        val_res = eval_on_split(model, val_logits, val_labels, device)
        test_res = eval_on_split(model, test_logits, test_labels, device)
        
        # Evaluate and collect single model results (VAL / TEST)
        print(f"\n=== Single Model Results ===")
        test_logits_per_model = test_logits  # [N_test, M, C]
        val_logits_per_model = val_logits    # [N_val,  M, C]
        
        print(f"\nTEST Set:")
        single_model_test_results = {}
        for m_idx, model_name in enumerate(model_names):
            single_test_logits = test_logits_per_model[:, m_idx, :]  # [N_test, C]
            single_test_res = eval_single_model(single_test_logits, test_labels)
            single_model_test_results[model_name] = single_test_res
            if 'auc' in single_test_res:
                print(f"  {model_name:20s}: AUC={single_test_res['auc']:.4f} ACC={single_test_res['acc']:.4f} F1={single_test_res['f1']:.4f}")
        
        print(f"\nVAL Set:")
        single_model_val_results = {}
        for m_idx, model_name in enumerate(model_names):
            single_val_logits = val_logits_per_model[:, m_idx, :]  # [N_val, C]
            single_val_res = eval_single_model(single_val_logits, val_labels)
            single_model_val_results[model_name] = single_val_res
            if 'auc' in single_val_res:
                print(f"  {model_name:20s}: AUC={single_val_res['auc']:.4f} ACC={single_val_res['acc']:.4f} F1={single_val_res['f1']:.4f}")
        
        # Update val_res / test_res with additional fields for per_split saving
        val_res.update({
            'split': 0,  # Patch-level has no explicit CV splits, use 0 as default
            'method': 'poe',
            'individual_models': single_model_val_results,
        })
        test_res.update({
            'split': 0,  # Patch-level has no splits, use 0 as default
            'method': 'poe',
            'individual_models': single_model_test_results,
        })

        if 'auc' in val_res:
            print(f"\n=== Fusion Model Results ===")
            print(f"VAL AUC={val_res['auc']:.4f} ACC={val_res['acc']:.4f} F1={val_res['f1']:.4f}")
        if 'auc' in test_res:
            print(f"TEST AUC={test_res['auc']:.4f} ACC={test_res['acc']:.4f} F1={test_res['f1']:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output directory
    out_dir = os.path.join(args.root, 'logits_fusion_results')
    os.makedirs(out_dir, exist_ok=True)

    # Helper function: convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj):
        """Recursively convert numpy arrays to Python lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    # Save summary results (backward compatibility)
    if val_res:
        val_fp = os.path.join(out_dir, 'poe_val_summary.json')
        # Remove fields not needed for summary
        val_summary = {k: v for k, v in val_res.items() 
                      if k not in ['split', 'individual_models']}
        with open(val_fp, 'w') as f:
            json.dump(convert_numpy_to_list(val_summary), f, indent=2, ensure_ascii=False)
        print(f"\nVAL results saved: {val_fp}")
        if 'auc' in val_res:
            print(f"Fusion VAL: AUC={val_res['auc']:.4f} ACC={val_res['acc']:.4f} F1={val_res['f1']:.4f}")
    
    if test_res:
        test_fp = os.path.join(out_dir, 'poe_test_summary.json')
        # Remove fields not needed for summary
        test_summary = {k: v for k, v in test_res.items() 
                       if k not in ['split', 'individual_models']}
        with open(test_fp, 'w') as f:
            json.dump(convert_numpy_to_list(test_summary), f, indent=2, ensure_ascii=False)
        print(f"TEST results saved: {test_fp}")
        if 'auc' in test_res:
            print(f"Fusion TEST: AUC={test_res['auc']:.4f} ACC={test_res['acc']:.4f} F1={test_res['f1']:.4f}")

    # Save per_split results (for plotting, even though there's only one split)
    # Format: list of results, each with split index
    all_val_results = [val_res] if val_res else []
    all_test_results = [test_res] if test_res else []
    
    if all_val_results:
        val_per_split_fp = os.path.join(out_dir, 'poe_val_per_split.json')
        val_results_serializable = convert_numpy_to_list(all_val_results)
        with open(val_per_split_fp, 'w') as f:
            json.dump(val_results_serializable, f, indent=2, ensure_ascii=False)
        print(f"VAL per-split results saved: {val_per_split_fp}")
    
    if all_test_results:
        test_per_split_fp = os.path.join(out_dir, 'poe_test_per_split.json')
        test_results_serializable = convert_numpy_to_list(all_test_results)
        with open(test_per_split_fp, 'w') as f:
            json.dump(test_results_serializable, f, indent=2, ensure_ascii=False)
        print(f"TEST per-split results saved: {test_per_split_fp}")

    # Save .npy files
    # Since patch-level has no splits, save directly to out_dir (not split_0 subdirectory)
    if test_res:
        try:
            # Extract data from test_res
            test_outputs = test_res.get('fused_logits', None)  # fused_logits
            test_probabilities = test_res.get('fused_probs', None)  # fused_probs
            test_predictions = test_res.get('preds', None)  # preds
            test_labels_save = test_res.get('labels', None)  # labels (should be in test_res from eval_on_split)

            # Save .npy files
            if test_outputs is not None:
                np.save(os.path.join(out_dir, 'test_outputs.npy'), test_outputs)
                print(f"Saved: {os.path.join(out_dir, 'test_outputs.npy')}")
            if test_predictions is not None:
                np.save(os.path.join(out_dir, 'test_predictions.npy'), test_predictions)
                print(f"Saved: {os.path.join(out_dir, 'test_predictions.npy')}")
            if test_labels_save is not None:
                np.save(os.path.join(out_dir, 'test_labels.npy'), test_labels_save)
                print(f"Saved: {os.path.join(out_dir, 'test_labels.npy')}")
            if test_probabilities is not None:
                np.save(os.path.join(out_dir, 'test_probabilities.npy'), test_probabilities)
                print(f"Saved: {os.path.join(out_dir, 'test_probabilities.npy')}")

            # Save test_results.npy (dictionary containing all information)
            test_results_dict = {
                'test_outputs': test_outputs,
                'test_predictions': test_predictions,
                'test_labels': test_labels_save,
                'test_probabilities': test_probabilities
            }
            np.save(os.path.join(out_dir, 'test_results.npy'), test_results_dict)
            print(f"Saved: {os.path.join(out_dir, 'test_results.npy')}")
        except Exception as e:
            print(f"Warning: Failed to save .npy files: {e}")
            import traceback
            traceback.print_exc()
    
    # Also save val .npy files (val and test use same dataset, but save for consistency)
    if val_res:
        try:
            val_outputs = val_res.get('fused_logits', None)
            val_probabilities = val_res.get('fused_probs', None)
            val_predictions = val_res.get('preds', None)
            val_labels_save = val_res.get('labels', None)  # labels (should be in val_res from eval_on_split)

            if val_outputs is not None:
                np.save(os.path.join(out_dir, 'val_outputs.npy'), val_outputs)
            if val_predictions is not None:
                np.save(os.path.join(out_dir, 'val_predictions.npy'), val_predictions)
            if val_labels_save is not None:
                np.save(os.path.join(out_dir, 'val_labels.npy'), val_labels_save)
            if val_probabilities is not None:
                np.save(os.path.join(out_dir, 'val_probabilities.npy'), val_probabilities)
        except Exception as e:
            print(f"Warning: Failed to save VAL .npy files: {e}")

if __name__ == '__main__':
    main()
