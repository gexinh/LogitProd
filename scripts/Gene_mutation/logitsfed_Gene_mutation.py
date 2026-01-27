#!/usr/bin/env python3
"""
POE Logit-based Fusion for Gene Mutation Prediction - GitHub Release Version

This script trains a lightweight gating network to fuse multiple expert models'
logits on slide-level gene mutation prediction tasks.

Key characteristics:
  - Only uses experts' logits to build gating features (no WSI embeddings used).
  - Uses Out-of-Fold (OOF) validation predictions from other folds to estimate
    an expert-correlation matrix R for regularization.
  - Produces per-split validation/test metrics and summary JSON outputs.

Expected directory structure:
  <root>/<model_name>/split_<k>/
    - train_features_labels.h5 (optional, only for ids/labels)
    - val_features_labels.h5
    - test_features_labels.h5
    - best_train_outputs.npy
    - best_val_outputs.npy
    - test_outputs.npy
    - (optional) best_train_labels.npy / best_val_labels.npy / test_labels.npy
  <root>/<model_name>/summary.json (optional, for reporting individual model metrics)

Usage example:
  python logitsfed_genemutation.py --dataset brca_lusc --target_gene ARID1A
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# import setup_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.loss import (
    compute_q_and_h,
    coral_loss_weighted,
    pair_margin_div_loss,
    cka_diversity_loss,
    entropy_loss
)

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
      fused_logits : [B, K] fused logits (compatible with NLLLoss)
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

    def forward(self, x: torch.Tensor, logits: torch.Tensor):
        raw = self.gate(x) / max(self.gate_temp, 1e-8)   # [B, M]
        weights = F.softmax(raw, dim=-1)                  # [B, M]
        log_p = F.log_softmax(logits, dim=-1)            # [B, M, K]
        fused_logits = torch.einsum('bm,bmk->bk', weights, log_p)  # [B, K] (log-prob like)
        return fused_logits, weights


# -------------------------------
# Dataset (ONLY logits + labels)
# -------------------------------
class SlideLogitsDataset(Dataset):
    """
    Minimal dataset for fusion training.
    We only need:
      - experts' logits: [N, M, C]
      - labels: [N]
    """
    def __init__(self, logits_per_model: np.ndarray, labels: np.ndarray = None):
        self.logits = torch.from_numpy(logits_per_model).float()
        self.labels = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self):
        return self.logits.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.logits[idx]
        return self.logits[idx], self.labels[idx]


# -------------------------------
# Utilities: loading + alignment
# -------------------------------
def _decode_ids(arr):
    """
    Decode slide IDs from H5 (bytes/object) into usable string arrays.
    """
    try:
        if isinstance(arr, np.ndarray) and arr.dtype.kind in ("S", "O"):
            return arr.astype(str)
        return arr
    except Exception:
        return arr


def load_h5_features_one_split(model_dir: str, split_idx: int):
    """
    Load {train,val,test}_features_labels.h5 for a single model and split.

    Note:
      - Features are not used in training, but we keep reading them here
        because slide_ids/labels may live in these H5 files.
    """
    split_dir = os.path.join(model_dir, f"split_{split_idx}")

    def _load(name):
        fp = os.path.join(split_dir, name)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

        with h5py.File(fp, "r") as f:
            feats = f["features"][:]
            labels = f["labels"][:] if "labels" in f else None
            slide_ids = _decode_ids(f["slide_ids"][:]) if "slide_ids" in f else np.array([str(i) for i in range(len(feats))])
        return feats, labels, slide_ids

    train_f = _load("train_features_labels.h5") if os.path.exists(os.path.join(split_dir, "train_features_labels.h5")) else None
    val_f = _load("val_features_labels.h5")
    test_f = _load("test_features_labels.h5")
    return train_f, val_f, test_f


def try_load_preds(split_dir: str):
    """
    Load per-split predicted logits saved by each expert model.
    """
    train_fp = os.path.join(split_dir, "best_train_outputs.npy")
    val_fp = os.path.join(split_dir, "best_val_outputs.npy")
    test_fp = os.path.join(split_dir, "test_outputs.npy")

    for fp in [train_fp, val_fp, test_fp]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

    return {
        "train": np.load(train_fp),
        "val": np.load(val_fp),
        "test": np.load(test_fp),
    }


def try_load_labels(split_dir: str):
    """
    Labels might be saved separately as .npy (preferred for compatibility).
    If not found, H5 labels will be used.
    """
    train_np = os.path.join(split_dir, "best_train_labels.npy")
    val_np = os.path.join(split_dir, "best_val_labels.npy")
    test_np = os.path.join(split_dir, "test_labels.npy")

    train_labels_np = np.load(train_np) if os.path.exists(train_np) else None
    val_labels_np = np.load(val_np) if os.path.exists(val_np) else None
    test_labels_np = np.load(test_np) if os.path.exists(test_np) else None
    return train_labels_np, val_labels_np, test_labels_np


def load_multi_model_features_and_probs(model_root_paths, split_idx):
    """
    Load all models' outputs for a split and align samples by common slide_ids.

    Returns:
      (train_feats, train_logits, train_ids, train_labels),
      (val_feats, val_logits, val_ids, val_labels),
      (test_feats, test_logits, test_ids, test_labels),
      model_names, n_models, n_classes, feat_dim
    """
    per_model = []
    model_names = []

    for p in model_root_paths:
        name = os.path.basename(p)
        try:
            train_f, val_f, test_f = load_h5_features_one_split(p, split_idx)
            (val_feats, val_labels_h5, val_ids) = val_f
            (test_feats, test_labels_h5, test_ids) = test_f

            if train_f is not None:
                (train_feats, train_labels_h5, train_ids) = train_f
            else:
                raise FileNotFoundError(f"{p}/split_{split_idx}/train_features_labels.h5 does not exist")

            split_dir = os.path.join(p, f"split_{split_idx}")
            train_labels_np, val_labels_np, test_labels_np = try_load_labels(split_dir)

            train_labels = train_labels_np if train_labels_np is not None else train_labels_h5
            val_labels = val_labels_np if val_labels_np is not None else val_labels_h5
            test_labels = test_labels_np if test_labels_np is not None else test_labels_h5

            preds_dict = try_load_preds(split_dir)
            train_logits = preds_dict["train"]
            val_logits = preds_dict["val"]
            test_logits = preds_dict["test"]

            per_model.append({
                "train_feats": train_feats,
                "train_labels": train_labels,
                "train_ids": train_ids,
                "train_logits": train_logits,
                "val_feats": val_feats,
                "val_labels": val_labels,
                "val_ids": val_ids,
                "val_logits": val_logits,
                "test_feats": test_feats,
                "test_labels": test_labels,
                "test_ids": test_ids,
                "test_logits": test_logits
            })
            model_names.append(name)

        except Exception as e:
            print(f"[skip] Failed to load model at {p}: {e}")
            continue

    if not per_model:
        raise RuntimeError("No model data loaded successfully.")

    def build_split(feat_key, id_key, logit_key, label_key):
        # Keep only slide_ids that exist for all models
        ids_sets = [set(m[id_key]) for m in per_model]
        common = sorted(list(set.intersection(*ids_sets)))

        feats_list, logits_list = [], []
        for m in per_model:
            id2idx = {sid: i for i, sid in enumerate(m[id_key])}
            idxs = [id2idx[sid] for sid in common]
            feats_list.append(m[feat_key][idxs])
            logits_list.append(m[logit_key][idxs])

        feats = np.stack(feats_list, axis=1)     # [N, M, D]
        logits = np.stack(logits_list, axis=1)   # [N, M, C]

        labels = None
        if per_model[0].get(label_key) is not None:
            labels_src = per_model[0][label_key]
            id2idx0 = {sid: i for i, sid in enumerate(per_model[0][id_key])}
            labels = labels_src[[id2idx0[sid] for sid in common]]

        return feats, logits, common, labels

    train_feats, train_logits, train_ids, train_labels = build_split("train_feats", "train_ids", "train_logits", "train_labels")
    val_feats, val_logits, val_ids, val_labels = build_split("val_feats", "val_ids", "val_logits", "val_labels")
    test_feats, test_logits, test_ids, test_labels = build_split("test_feats", "test_ids", "test_logits", "test_labels")

    n_models = train_feats.shape[1]
    n_classes = train_logits.shape[2]
    feat_dim = train_feats.shape[2]
    return (
        (train_feats, train_logits, train_ids, train_labels),
        (val_feats, val_logits, val_ids, val_labels),
        (test_feats, test_logits, test_ids, test_labels),
        model_names,
        n_models,
        n_classes,
        feat_dim
    )


# -------------------------------
# Gating feature extraction
# -------------------------------
@torch.no_grad()
def logit_feat_extraction(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Construct gating feature x (only based on experts' logits):
      - per-expert: MSP, Margin, Entropy → each dimension [B, M]
      - cross-expert: Avg-Entropy (mean entropy), Mutual Information (H(mean) - mean H) → [B,1]×2
    Returns:
      x: [B, 3*M + 2]
    """
    # logits: [B, M, K]
    B, M, K = logits.shape
    # stable softmax
    p = F.softmax(logits, dim=-1)                                   # [B, M, K]
    logp = (p + eps).log()

    # per-expert MSP & Margin
    top2 = torch.topk(p, k=min(2, K), dim=-1).values                # [B, M, min(2,K)]
    msp   = top2[..., 0]                                            # [B, M]
    if K >= 2:
        margin = top2[..., 0] - top2[..., 1]                        # [B, M]
    else:
        margin = torch.zeros_like(msp)

    # per-expert Entropy
    ent = -(p * logp).sum(dim=-1)                                   # [B, M]

    # cross-expert: average entropy & mutual information
    avg_entropy = ent.mean(dim=1, keepdim=True)                     # [B, 1]
    p_bar = p.mean(dim=1)                                           # [B, K]
    H_bar = -(p_bar * (p_bar + eps).log()).sum(dim=-1, keepdim=True)  # [B, 1]
    mutual_info = H_bar - avg_entropy                               # [B, 1]

    # concatenate to [B, 3M + 2]
    x = torch.cat([msp, margin, ent], dim=-1)                       # [B, 3M]
    x = torch.cat([x, avg_entropy, mutual_info], dim=-1)            # [B, 3M + 2]
    return x


def build_oof_train_from_other_folds(model_root_paths, cur_split, n_splits):
    """
    Construct Out-of-Fold (OOF) data for estimating correlation matrix R.
    For each split != cur_split, collect its validation logits/labels.
    """
    oof_feats, oof_logits, oof_labels = [], [], []
    ref_model_names = None
    ref_meta = None  # (M, C, D)

    for s in range(n_splits):
        if s == cur_split:
            continue

        (train_s, val_s, test_s,
         model_names_s, n_models_s, n_classes_s, feat_dim_s) = load_multi_model_features_and_probs(
            model_root_paths, s
        )

        val_feats_s, val_logits_s, val_ids_s, val_labels_s = val_s
        if val_labels_s is None:
            continue

        if ref_model_names is None:
            ref_model_names = model_names_s
            ref_meta = (n_models_s, n_classes_s, feat_dim_s)
        else:
            if model_names_s != ref_model_names:
                raise RuntimeError(f"Model order mismatch at split {s}")
            if (n_models_s, n_classes_s, feat_dim_s) != ref_meta:
                raise RuntimeError(f"Meta mismatch at split {s}")

        oof_feats.append(val_feats_s)
        oof_logits.append(val_logits_s)
        oof_labels.append(val_labels_s)

    if not oof_feats:
        raise RuntimeError("No OOF data collected from other folds.")

    oof_feats = np.concatenate(oof_feats, axis=0)
    oof_logits = np.concatenate(oof_logits, axis=0)
    oof_labels = np.concatenate(oof_labels, axis=0)

    return oof_feats, oof_logits, oof_labels, ref_model_names, *ref_meta


# -------------------------------
# R matrix + penalty
# -------------------------------
@torch.no_grad()
def compute_R_from_oof(oof_logits: torch.Tensor,
                       oof_labels: torch.Tensor,
                       kind: str = "nll",
                       shrink: float = 0.1,
                       pos_only: bool = True,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    oof_logits: [N_oof, M, C], per-model temperature calibration recommended for comparability
    oof_labels: [N_oof]
    Returns R: [M, M] (PSD / symmetric / optionally non-negative only)
    """
    oof_logits = oof_logits.to(dtype=torch.float32)
    device = oof_logits.device
    N, M, C = oof_logits.shape

    if kind == "nll":
        p = F.softmax(oof_logits, dim=-1)                  # [N,M,C]
        y = oof_labels.to(torch.long).view(-1, 1, 1).expand(-1, M, 1)     # [N,M,1]
        nll = -torch.log(p.gather(-1, y).clamp_min(eps)).squeeze(-1)  # [N,M]
        X = nll                                            # larger error is worse
    else:
        raise ValueError("kind must be 'nll'")

    # Standardize within models (avoid different scales)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd                                      # [N,M]

    # Correlation matrix (unit variance)
    R = (Z.t() @ Z) / (Z.shape[0] - 1)                     # [M,M]
    R = (R + R.t()) * 0.5                                  # symmetrize

    # Shrink to diagonal to reduce small-sample noise
    if shrink > 0:
        diag_mean = torch.diag(R).mean()
        I = torch.eye(M, device=device)
        R = (1 - shrink) * R + shrink * diag_mean * I

    # PSD projection (clamp negative eigenvalues to 0)
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(0.0)
    R_psd = (eigvecs * eigvals.clamp_min(0.0)) @ eigvecs.t()

    # Only penalize positive correlations (leave negative correlations unchanged), and set diagonal to 0 (no self-correlation penalty)
    if pos_only:
        R_psd = torch.clamp(R_psd, min=0.0)
    R_psd.fill_diagonal_(0.0)

    return R_psd


def func2(weights: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    penalty (mode A):
      - A: global correlation penalty wᵀ R w (R from OOF error correlation/covariance, PSD/symmetric, shrinkage recommended)

    Parameters:
      weights: [B, M], expert weights after softmax
      R      : [M, M], recommended to be symmetrized and non-negative clipped
    Returns:
      scalar penalty (torch.Tensor)
    """
    assert weights.dim() == 2, "weights shape must be [B, M]"
    B, M = weights.shape
    assert R is not None and R.shape == (M, M), "R must be [M,M] for mode A"
    
    # Symmetrize + non-negative clipping (only penalize positive correlations; negative correlations are complementary, not penalized)
    R_use = 0.5 * (R + R.t())
    R_use = torch.clamp(R_use, min=0.0)
    # batch average of wᵀ R w
    quad = torch.einsum("bm,mn,bn->b", weights, R_use, weights)   # [B]
    return quad.mean()


# -------------------------------
# Train with val
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
    Two-stage training (this version only shows CE + penalty; you can add warmup Align/Diversity at outer level if needed)
    - Training stage: optimize total_loss = CE + lambda_l * penalty
    - Validation stage: use only CE as early stopping metric, but also print CE and penalty for observation
    """
    lr = lr
    weight_decay = 1e-4

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    ce = nn.NLLLoss()

    best_state = None
    best_val_loss = float("inf")
    patience, counter = 50, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_tot, train_ce_accum, train_pen_accum = 0.0, 0.0, 0.0

        for logits, labels in train_loader:
            logits = logits.to(device)     # [B, M, K]
            labels = labels.to(device)     # [B]
            x = logit_feat_extraction(logits)

            fused_logits, weights = model(x, logits)

            # penalty (if your func2 needs logits/R etc., pass them here)
            pen = func2(weights, R=R)

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

        # ---------------- Validation ----------------
        model.eval()
        val_ce_accum, val_pen_accum = 0.0, 0.0
        with torch.no_grad():
            for logits, labels in val_loader:
                logits = logits.to(device)
                labels = labels.to(device)
                x = logit_feat_extraction(logits)

                fused_logits, weights = model(x, logits)

                loss_ce = ce(fused_logits, labels)
                # penalty for observation only (not used for early stopping/model selection)
                pen = func2(weights, R=R)

                val_ce_accum  += loss_ce.item()
                val_pen_accum += float(pen.detach().cpu())

        n_val = max(1, len(val_loader))
        avg_val_ce  = val_ce_accum / n_val
        avg_val_pen = val_pen_accum / n_val

        # Keep original logic: validation early stopping only uses CE
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
# Eval
# -------------------------------
@torch.no_grad()
def eval_on_split(model: nn.Module, logits: np.ndarray, labels: np.ndarray, device: torch.device, weight_threshold: float = 0.0):
    model.eval()
    B = logits.shape[0]
    batch = 64
    outs = []
    for i in range(0, B, batch):
        lm = torch.from_numpy(logits[i:i+batch]).float().to(device)
        x = logit_feat_extraction(lm)
        fused_logits, w = model(x, lm)
        outs.append(fused_logits.cpu().numpy())

    fused_logits = np.concatenate(outs, axis=0)
    fused_probs = np.exp(fused_logits)  # Convert log-prob to prob
    preds = np.argmax(fused_probs, axis=1)
    results = {}
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
        results.update({'auc': float(auc), 'acc': float(acc), 'f1': float(f1)})
    return results

@torch.no_grad()
def print_model_mean_weights(model: nn.Module, logits: np.ndarray, model_names: list, device: torch.device, title: str = "Weights"):
    model.eval()
    B = logits.shape[0]
    batch = 64
    total_weights = None
    n_seen = 0
    for i in range(0, B, batch):
        lm = torch.from_numpy(logits[i:i+batch]).float().to(device)
        x = logit_feat_extraction(lm)
        _, w = model(x, lm)
        if w.dim() == 3:
            batch_sum = w.detach().cpu().sum(dim=0)
            if total_weights is None:
                total_weights = batch_sum
            else:
                total_weights += batch_sum
            n_seen += w.shape[0]
        elif w.dim() == 2:
            batch_sum = w.detach().cpu().sum(dim=0)
            if total_weights is None: 
                total_weights = batch_sum
            else:
                total_weights += batch_sum
            n_seen += w.shape[0]
        else:
            raise ValueError("Unexpected weight shape")
    if n_seen == 0:
        print(f"{title}: no samples to compute weights")
        return
    mean_weights = (total_weights / float(n_seen)).numpy()
    if mean_weights.ndim == 2:
        C, M = mean_weights.shape
        print(f"===== {title}: Per-Class Mean Weights =====")
        for c in range(C):
            weights = mean_weights[c]
            order = np.argsort(weights)[::-1]
            print(f"Class {c}:")
            for rank, idx in enumerate(order, 1):
                print(f"  {rank:2d}. {model_names[idx]:<15} {weights[idx]:.4f}")
    elif mean_weights.ndim == 1:
        M = mean_weights.shape[0]
        print(f"===== {title}: Global Mean Weights =====")
        order = np.argsort(mean_weights)[::-1]
        for rank, idx in enumerate(order, 1):
            print(f"  {rank:2d}. {model_names[idx]:<15} {mean_weights[idx]:.4f}")
    else:
        raise ValueError("Unexpected mean weight ndim")
    return mean_weights


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Per-Class or Global Weighted Fusion (train on train, validate on val, test on test)')
    parser.add_argument('--dataset', type=str, default='brca_lusc', choices=['brca_lusc'])
    parser.add_argument('--target_gene', type=str, default='ARID1A', choices=['TP53', 'PIK3CA', 'PTEN', 'ARID1A', 'NF1'])
    parser.add_argument('--root', type=str, help='root containing per-model dirs with features and predictions')
    parser.add_argument(
        '--method',
        type=str,
        default='poe',
        choices=['poe']
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['virchow2', 'conch_v15', 'hoptimus1', 'gigapath', 'uni_v2', 'phikon_v2'],
        help='List of model folder names under <root>.'
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)  
    args = parser.parse_args()

    model_root_paths = [os.path.join(args.root, d) for d in args.models if os.path.exists(os.path.join(args.root, d))]
    if not model_root_paths:
        raise RuntimeError(f'No valid model directories found under root={args.root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] dataset={args.dataset} target_gene={args.target_gene} root={args.root}')
    print(f'[info] models={args.models}')
    print(f'[info] device={device}')

    out_dir = os.path.join(args.root, 'logits_based_fusion_poe')
    os.makedirs(out_dir, exist_ok=True)

    all_val_results = []
    all_test_results = []

    for split_idx in range(args.n_splits):
        try:
            (train_feats, train_logits, train_ids, train_labels), \
            (val_feats, val_logits, val_ids, val_labels), \
            (test_feats, test_logits, test_ids, test_labels), \
            model_names, n_models, n_classes, feat_dim = load_multi_model_features_and_probs(model_root_paths, split_idx)

            # === Build OOF training set (merge Val from other folds) ===
            # If there is only one split (e.g., PANDA), use current split's train data as OOF
            if args.n_splits == 1:
                oof_feats = train_feats
                oof_logits = train_logits
                oof_labels = train_labels
                ref_names = model_names
                ref_M = n_models
                ref_C = n_classes
                ref_D = feat_dim
            else:
                (oof_feats, oof_logits, oof_labels,
                 ref_names, ref_M, ref_C, ref_D) = build_oof_train_from_other_folds(
                    model_root_paths, cur_split=split_idx, n_splits=args.n_splits
                )

            R = compute_R_from_oof(
                torch.from_numpy(oof_logits).to(device),
                torch.from_numpy(oof_labels).long().to(device),
                kind="nll", shrink=0.1
            )
            # Consistency check again
            assert ref_M == n_models and ref_C == n_classes and ref_D == feat_dim, \
                "OOF meta and current fold meta mismatch"
            assert ref_names == model_names, "OOF model order != current fold model order"

            train_ds = SlideLogitsDataset(train_logits, labels=train_labels)
            val_ds = SlideLogitsDataset(val_logits, labels=val_labels)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

            model = POE(
                n_models=n_models,
                in_dim=n_models * 3 + 2,
                hidden=0,
            ).to(device)

            model = train_with_val(
                model, train_loader, val_loader, device,
                num_epochs=args.num_epochs, lr=args.lr, R=R
            )

            print_model_mean_weights(model, test_logits, model_names, device, title=f"Split {split_idx} Weights (Test)")

            val_res = eval_on_split(model, val_logits, val_labels, device, weight_threshold=0.0)
            val_res.update({
                'split': split_idx, 
                'method': args.method
            })
            all_val_results.append(val_res)

            test_res = eval_on_split(model, test_logits, test_labels, device, weight_threshold=0.0)
            test_res.update({
                'split': split_idx, 
                'method': args.method
            })
            all_test_results.append(test_res)

            if 'auc' in val_res:
                print(f"[split {split_idx}] VAL AUC={val_res['auc']:.4f} ACC={val_res['acc']:.4f} F1={val_res['f1']:.4f}")
            else:
                print(f"[split {split_idx}] VAL (no labels to score)")
            if 'auc' in test_res:
                print(f"[split {split_idx}] TEST AUC={test_res['auc']:.4f} ACC={test_res['acc']:.4f} F1={test_res['f1']:.4f}")
            else:
                print(f"[split {split_idx}] TEST (no labels to score)")
        except Exception as e:
            print(f"split {split_idx} failed: {e}")
            continue

    def summarize(results):
        if not results:
            return None
        aucs = [r['auc'] for r in results if 'auc' in r]
        accs = [r['acc'] for r in results if 'acc' in r]
        f1s = [r['f1'] for r in results if 'f1' in r]
        
        summary = {
            'method': args.method,
            'splits': len(results),
            'mean_auc': float(np.mean(aucs)) if aucs else None,
            'std_auc': float(np.std(aucs)) if aucs else None,
            'mean_acc': float(np.mean(accs)) if accs else None,
            'std_acc': float(np.std(accs)) if accs else None,
            'mean_f1': float(np.mean(f1s)) if f1s else None,
            'std_f1': float(np.std(f1s)) if f1s else None,
        }
        
        return summary

    val_summary = summarize(all_val_results)
    test_summary = summarize(all_test_results)

    # Save per-split data
    if all_val_results:
        val_per_split = {r['split']: {'val_auc': r.get('auc'), 'val_acc': r.get('acc'), 'val_f1': r.get('f1')} 
                        for r in all_val_results if 'split' in r}
        val_per_split_file = os.path.join(out_dir, f'{args.method}_val_per_split.json')
        with open(val_per_split_file, 'w') as f:
            json.dump(val_per_split, f, indent=2, ensure_ascii=False)
    
    if all_test_results:
        test_per_split = {r['split']: {'test_auc': r.get('auc'), 'test_acc': r.get('acc'), 'test_f1': r.get('f1')} 
                         for r in all_test_results if 'split' in r}
        test_per_split_file = os.path.join(out_dir, f'{args.method}_test_per_split.json')
        with open(test_per_split_file, 'w') as f:
            json.dump(test_per_split, f, indent=2, ensure_ascii=False)

    if val_summary:
        val_fp = os.path.join(out_dir, f'{args.method}_val_summary.json')
        with open(val_fp, 'w') as f:
            json.dump(val_summary, f, indent=2, ensure_ascii=False)
        print(f"VAL results saved: {val_fp}")
        val_auc_mean = f"{val_summary['mean_auc']:.4f}" if val_summary.get('mean_auc') is not None else "NA"
        val_auc_std = f"{val_summary['std_auc']:.4f}" if val_summary.get('std_auc') is not None else "NA"
        val_acc_mean = f"{val_summary['mean_acc']:.4f}" if val_summary.get('mean_acc') is not None else "NA"
        val_acc_std = f"{val_summary['std_acc']:.4f}" if val_summary.get('std_acc') is not None else "NA"
        val_f1_mean = f"{val_summary['mean_f1']:.4f}" if val_summary.get('mean_f1') is not None else "NA"
        val_f1_std = f"{val_summary['std_f1']:.4f}" if val_summary.get('std_f1') is not None else "NA"
        print(f"VAL summary (5-fold): splits={val_summary['splits']} | AUC={val_auc_mean}±{val_auc_std} | ACC={val_acc_mean}±{val_acc_std} | F1={val_f1_mean}±{val_f1_std}")
    if test_summary:
        test_fp = os.path.join(out_dir, f'{args.method}_test_summary.json')
        with open(test_fp, 'w') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        print(f"TEST results saved: {test_fp}")
        test_auc_mean = f"{test_summary['mean_auc']:.4f}" if test_summary.get('mean_auc') is not None else "NA"
        test_auc_std = f"{test_summary['std_auc']:.4f}" if test_summary.get('std_auc') is not None else "NA"
        test_acc_mean = f"{test_summary['mean_acc']:.4f}" if test_summary.get('mean_acc') is not None else "NA"
        test_acc_std = f"{test_summary['std_acc']:.4f}" if test_summary.get('std_acc') is not None else "NA"
        test_f1_mean = f"{test_summary['mean_f1']:.4f}" if test_summary.get('mean_f1') is not None else "NA"
        test_f1_std = f"{test_summary['std_f1']:.4f}" if test_summary.get('std_f1') is not None else "NA"
        print(f"TEST summary (5-fold): splits={test_summary['splits']} | AUC={test_auc_mean}±{test_auc_std} | ACC={test_acc_mean}±{test_acc_std} | F1={test_f1_mean}±{test_f1_std}")


if __name__ == '__main__':
    main()