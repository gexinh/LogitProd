#!/usr/bin/env python3
"""
POE Logit-based Fusion (Train/Val/Test) - GitHub Release Version (Survival)

This script trains a lightweight gating network to fuse multiple expert models'
logits on slide-level survival prediction tasks (discrete-time hazards).

Key characteristics:
  - Only uses experts' hazard logits to build gating features (no WSI embeddings used).
  - Uses Out-of-Fold (OOF) validation predictions from other folds to estimate
    an expert-correlation matrix R for regularization.
  - Produces per-split validation/test metrics and summary JSON outputs.

Expected directory structure:
  <root>/<model_name>/split_<k>/
    - train_features_labels.h5 (optional, used only for ids/labels)
    - val_features_labels.h5
    - test_features_labels.h5
    - best_train_preds.npy
    - best_val_preds.npy
    - test_preds.npy
    - (optional) best_train_labels.npy / best_val_labels.npy / test_labels.npy

Usage example:
  python train_poe_survival_fusion.py --dataset kirc --root ./outputs/kirc
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
from sksurv.metrics import concordance_index_censored

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Defaults (hard-coded)
# NOTE: Do NOT change survival logic/params here.
# -------------------------------
DEFAULT_LAMBDA_L = 0.1      # decorrelation penalty weight (fixed)
DEFAULT_SHRINK = 0.1        # shrinkage for R
DEFAULT_POS_ONLY = True     # clamp R >= 0
DEFAULT_GATE_TEMP = 1.0     # softmax temperature for POE gate


# -------------------------------
# Survival: Distribution conversion
# -------------------------------
@torch.no_grad()
def logits_to_full_event_dist(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert hazard logits -> full event distribution p(t) with tail bin.

    logits : [B, M, T] (hazard logits)
    return : [B, M, T+1] (event prob at each time + tail)
    """
    h = torch.sigmoid(logits)  # [B, M, T]
    one_minus_h = (1.0 - h).clamp_min(eps)
    S_prefix = torch.cumprod(one_minus_h, dim=-1)  # S_t = Π_{j<=t} (1-h_j)

    S_prev = torch.ones_like(S_prefix)
    S_prev[..., 1:] = S_prefix[..., :-1]          # S_{t-1}

    p_event = (S_prev * h).clamp_min(eps)         # [B, M, T]
    p_tail = S_prefix[..., -1].clamp_min(eps)     # [B, M]
    p_full = torch.cat([p_event, p_tail.unsqueeze(-1)], dim=-1)  # [B, M, T+1]
    return p_full


@torch.no_grad()
def full_event_dist_to_hazards(p_full: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert full event distribution -> hazards.

    p_full : [B, T+1]
    return : [B, T]
    """
    p_event = p_full[..., :-1].clamp_min(eps)  # [B, T]
    cdf = torch.cumsum(p_event, dim=-1)        # [B, T]

    S_prev = torch.ones_like(cdf)
    S_prev[..., 1:] = (1.0 - cdf[..., :-1]).clamp_min(eps)

    h = (p_event / S_prev).clamp(min=eps, max=1.0 - eps)
    return h


# -------------------------------
# Model: POE (hazard-logit-based gating)
# -------------------------------
class POESurvival(nn.Module):
    """
    Product-of-Experts style fusion for survival.

    Inputs:
      x      : [B, D] gating feature vector extracted from experts' hazard logits
      logits : [B, M, T] experts' hazard logits

    Outputs:
      fused_hazards : [B, T] fused hazards in (0, 1)
      weights       : [B, M] sample-wise expert weights (Softmax normalized)
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
        raw = self.gate(x) / max(self.gate_temp, 1e-8)  # [B, M]
        weights = F.softmax(raw, dim=-1)                # [B, M]

        # POE on full event distribution (log domain)
        p_full = logits_to_full_event_dist(logits)      # [B, M, T+1]
        logp = (p_full + 1e-12).log()                   # [B, M, T+1]

        fused_logp = torch.einsum('bm,bmk->bk', weights, logp)  # [B, T+1]
        fused_p_full = F.softmax(fused_logp, dim=-1)            # [B, T+1]
        fused_hazards = full_event_dist_to_hazards(fused_p_full)  # [B, T]
        return fused_hazards, weights


class POESurvivalPerTimeBin(nn.Module):
    """
    POE fusion with one independent gate network per time bin (including tail bin).

    Inputs:
      x      : [B, D] gating feature vector extracted from experts' hazard logits
      logits : [B, M, T] experts' hazard logits

    Outputs:
      fused_hazards : [B, T] fused hazards in (0, 1)
      weights_mean  : [B, M] mean expert weights across bins (for penalty compatibility)
    """
    def __init__(
        self,
        in_dim: int,
        n_models: int,
        n_bins: int,
        hidden: int = 0,
        gate_temp: float = 1.0,
        init_uniform: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_bins = n_bins
        self.gate_temp = gate_temp

        if hidden and hidden > 0:
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_dim, hidden, bias=True),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, n_models, bias=bias),
                ) for _ in range(n_bins)
            ])
        else:
            self.gates = nn.ModuleList([
                nn.Linear(in_dim, n_models, bias=bias) for _ in range(n_bins)
            ])

        if init_uniform:
            for g in self.gates:
                if isinstance(g, nn.Linear):
                    nn.init.zeros_(g.weight)
                    if bias:
                        nn.init.zeros_(g.bias)
                else:
                    nn.init.zeros_(g[-1].weight)
                    if bias:
                        nn.init.zeros_(g[-1].bias)

    def forward(self, x: torch.Tensor, logits: torch.Tensor):
        # one gate per bin -> weights: [B, T+1, M]
        w_list = []
        for k in range(self.n_bins):
            raw_k = self.gates[k](x) / max(self.gate_temp, 1e-8)  # [B, M]
            w_k = F.softmax(raw_k, dim=-1)                        # [B, M]
            w_list.append(w_k)
        weights = torch.stack(w_list, dim=1)                      # [B, T+1, M]

        # POE on full event distribution in log-domain, per bin
        p_full = logits_to_full_event_dist(logits)                # [B, M, T+1]
        logp = (p_full + 1e-12).log()                             # [B, M, T+1]
        fused_logp = torch.einsum("btm,bmt->bt", weights, logp)   # [B, T+1]
        fused_p_full = F.softmax(fused_logp, dim=-1)              # [B, T+1]
        fused_hazards = full_event_dist_to_hazards(fused_p_full)  # [B, T]

        # keep penalty interface expecting [B, M]
        weights_mean = weights.mean(dim=1)                        # [B, M]
        return fused_hazards, weights_mean


# -------------------------------
# Dataset (ONLY logits + labels)
# -------------------------------
class SurvivalLogitDataset(Dataset):
    """
    Minimal dataset for fusion training.
    We only need:
      - experts' hazard logits: [N, M, T]
      - labels: [N, 3] -> [Y, censored_flag, survival_time]
    """
    def __init__(self, preds_per_model: np.ndarray, labels: np.ndarray):
        self.preds = torch.from_numpy(preds_per_model).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return self.preds.shape[0]

    def __getitem__(self, idx):
        return self.preds[idx], self.labels[idx]


# -------------------------------
# Utilities: loading + alignment
# -------------------------------
def _decode_ids(arr):
    """
    Decode slide IDs from H5 (bytes/object) into usable string arrays.
    """
    try:
        if isinstance(arr, np.ndarray) and arr.dtype.kind in ('S', 'O'):
            return arr.astype(str)
        return arr
    except Exception:
        return arr


def load_h5_ids_labels_one_split(model_dir: str, split_idx: int):
    """
    Load only slide_ids + labels from {train,val,test}_features_labels.h5 for one model/split.

    NOTE:
      - We do NOT load features anymore (to reduce memory usage).
    """
    split_dir = os.path.join(model_dir, f'split_{split_idx}')

    def _load(name):
        fp = os.path.join(split_dir, name)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

        with h5py.File(fp, 'r') as f:
            labels = f['labels'][:] if 'labels' in f else None
            if 'slide_ids' in f:
                slide_ids = _decode_ids(f['slide_ids'][:])
            else:
                # fallback: stable index-based ids
                n = len(labels) if labels is not None else 0
                slide_ids = np.array([str(i) for i in range(n)])
        return labels, slide_ids

    train_f = _load('train_features_labels.h5') if os.path.exists(os.path.join(split_dir, 'train_features_labels.h5')) else None
    val_f = _load('val_features_labels.h5')
    test_f = _load('test_features_labels.h5')
    return train_f, val_f, test_f


def try_load_preds(split_dir: str):
    """
    Load per-split predicted hazard logits saved by each expert model.
    """
    train_fp = os.path.join(split_dir, 'best_train_preds.npy')
    val_fp = os.path.join(split_dir, 'best_val_preds.npy')
    test_fp = os.path.join(split_dir, 'test_preds.npy')

    for fp in [train_fp, val_fp, test_fp]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

    return {
        'train': np.load(train_fp),
        'val': np.load(val_fp),
        'test': np.load(test_fp),
    }


def try_load_labels(split_dir: str):
    """
    Labels might be saved separately as .npy (preferred for compatibility).
    If not found, H5 labels will be used.
    """
    train_np = os.path.join(split_dir, 'best_train_labels.npy')
    val_np = os.path.join(split_dir, 'best_val_labels.npy')
    test_np = os.path.join(split_dir, 'test_labels.npy')

    train_labels_np = np.load(train_np) if os.path.exists(train_np) else None
    val_labels_np = np.load(val_np) if os.path.exists(val_np) else None
    test_labels_np = np.load(test_np) if os.path.exists(test_np) else None
    return train_labels_np, val_labels_np, test_labels_np


def load_multi_model_preds_and_labels(model_root_paths, split_idx):
    """
    Load all models' outputs for a split and align samples by common slide_ids.

    Returns:
      (train_logits, train_ids, train_labels),
      (val_logits, val_ids, val_labels),
      (test_logits, test_ids, test_labels),
      model_names, n_models, n_bins
    """
    per_model = []
    model_names = []

    for p in model_root_paths:
        name = os.path.basename(p)
        try:
            train_f, val_f, test_f = load_h5_ids_labels_one_split(p, split_idx)

            (val_labels_h5, val_ids) = val_f
            (test_labels_h5, test_ids) = test_f

            if train_f is not None:
                (train_labels_h5, train_ids) = train_f
            else:
                raise FileNotFoundError(f"{p}/split_{split_idx}/train_features_labels.h5 does not exist")

            split_dir = os.path.join(p, f'split_{split_idx}')
            train_labels_np, val_labels_np, test_labels_np = try_load_labels(split_dir)

            train_labels = train_labels_np if train_labels_np is not None else train_labels_h5
            val_labels = val_labels_np if val_labels_np is not None else val_labels_h5
            test_labels = test_labels_np if test_labels_np is not None else test_labels_h5

            preds_dict = try_load_preds(split_dir)
            train_preds = preds_dict['train']
            val_preds = preds_dict['val']
            test_preds = preds_dict['test']

            per_model.append({
                'train_labels': train_labels,
                'train_ids': train_ids,
                'train_preds': train_preds,

                'val_labels': val_labels,
                'val_ids': val_ids,
                'val_preds': val_preds,

                'test_labels': test_labels,
                'test_ids': test_ids,
                'test_preds': test_preds,
            })
            model_names.append(name)

        except Exception as e:
            print(f"[skip] Failed to load model at {p}: {e}")
            continue

    if not per_model:
        raise RuntimeError("No model data loaded successfully.")

    def build_split(id_key, pred_key, label_key):
        ids_sets = [set(m[id_key]) for m in per_model]
        common = sorted(list(set.intersection(*ids_sets)))

        preds_list = []
        for m in per_model:
            id2idx = {sid: i for i, sid in enumerate(m[id_key])}
            idxs = [id2idx[sid] for sid in common]
            preds_list.append(m[pred_key][idxs])

        preds = np.stack(preds_list, axis=1)  # [N, M, T]

        labels = None
        if per_model[0].get(label_key) is not None:
            labels_src = per_model[0][label_key]
            id2idx0 = {sid: i for i, sid in enumerate(per_model[0][id_key])}
            labels = labels_src[[id2idx0[sid] for sid in common]]

        return preds, common, labels

    train_preds, train_ids, train_labels = build_split('train_ids', 'train_preds', 'train_labels')
    val_preds, val_ids, val_labels = build_split('val_ids', 'val_preds', 'val_labels')
    test_preds, test_ids, test_labels = build_split('test_ids', 'test_preds', 'test_labels')

    n_models = train_preds.shape[1]
    n_bins = train_preds.shape[2]
    return (
        (train_preds, train_ids, train_labels),
        (val_preds, val_ids, val_labels),
        (test_preds, test_ids, test_labels),
        model_names, n_models, n_bins
    )


def build_oof_from_other_folds(model_root_paths, cur_split, n_splits):
    """
    Construct Out-of-Fold (OOF) data for estimating correlation matrix R.
    For each split != cur_split, collect its validation hazard logits/labels.
    """
    oof_logits, oof_labels = [], []
    ref_model_names = None
    ref_meta = None  # (M, T)

    for s in range(n_splits):
        if s == cur_split:
            continue

        (train_s, val_s, test_s,
         model_names_s, n_models_s, n_bins_s) = load_multi_model_preds_and_labels(model_root_paths, s)

        val_logits_s, _, val_labels_s = val_s
        if val_labels_s is None:
            continue

        if ref_model_names is None:
            ref_model_names = model_names_s
            ref_meta = (n_models_s, n_bins_s)
        else:
            if model_names_s != ref_model_names:
                raise RuntimeError(f"Model order mismatch at split {s}")
            if (n_models_s, n_bins_s) != ref_meta:
                raise RuntimeError(f"Meta mismatch at split {s}")

        oof_logits.append(val_logits_s)
        oof_labels.append(val_labels_s)

    if not oof_logits:
        raise RuntimeError("No OOF data collected from other folds.")

    oof_logits = np.concatenate(oof_logits, axis=0)
    oof_labels = np.concatenate(oof_labels, axis=0)
    return oof_logits, oof_labels, ref_model_names, *ref_meta


# -------------------------------
# Gating feature extraction (from hazard logits)
# -------------------------------
@torch.no_grad()
def survival_pred_feat_extraction(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Build gating feature vector x from experts' hazard logits only.

    Returns:
      x: [B, 3*M + 2]
        - per-expert: MSP(event-dist max), margin(top1-top2), entropy
        - global: avg_entropy, mutual_info
    """
    B, M, T = logits.shape
    p_full = logits_to_full_event_dist(logits, eps=eps)  # [B, M, T+1]
    logp = (p_full + eps).log()

    top2 = torch.topk(p_full, k=min(2, T + 1), dim=-1).values
    msp = top2[..., 0]  # [B, M]
    margin = top2[..., 0] - top2[..., 1] if (T + 1) >= 2 else torch.zeros_like(msp)

    ent = -(p_full * logp).sum(dim=-1)  # [B, M]
    avg_entropy = ent.mean(dim=1, keepdim=True)

    p_bar = p_full.mean(dim=1)  # [B, T+1]
    H_bar = -(p_bar * (p_bar + eps).log()).sum(dim=-1, keepdim=True)
    mutual_info = H_bar - avg_entropy

    x = torch.cat([msp, margin, ent], dim=-1)  # [B, 3M]
    x = torch.cat([x, avg_entropy, mutual_info], dim=-1)  # [B, 3M+2]
    return x


# -------------------------------
# Survival loss & metrics
# -------------------------------
def survival_loss(hazards: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4,
                  eps: float = 1e-7, reduction: str = 'mean'):
    """
    hazards: [B, T]
    labels : [B, 3] -> [Y, censored_flag, survival_time]
      - Y is 1-based discrete bin index
      - censored_flag = 1 means censored, 0 means event
    """
    if labels is None:
        return torch.tensor(0.0, device=hazards.device)

    B = hazards.size(0)
    T = hazards.size(1)

    Y = labels[:, 0].long().clamp(min=1, max=T)  # 1-based
    c = labels[:, 1].float()

    h = hazards.clamp(min=eps, max=1.0 - eps)
    S = torch.cumprod(1.0 - h, dim=1)
    ones = torch.ones((B, 1), dtype=S.dtype, device=S.device)
    S_pad = torch.cat([ones, S], dim=1)  # [B, T+1]

    hY = torch.gather(h, 1, (Y - 1).clamp(min=0, max=T-1).unsqueeze(1)).squeeze(1)
    S_before = torch.gather(S_pad, 1, (Y - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    S_atY = torch.gather(S_pad, 1, Y.clamp(max=T).unsqueeze(1)).squeeze(1)

    uncens = - (1.0 - c) * (torch.log(S_before.clamp(min=eps)) + torch.log(hY.clamp(min=eps)))
    cens = - c * torch.log(S_atY.clamp(min=eps))
    neg_l = cens + uncens
    loss = (1.0 - alpha) * neg_l + alpha * uncens

    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    raise ValueError(f"reduction must be 'none' or 'mean', got {reduction}")


def safe_concordance_index(times, censorship, risks):
    times = np.asarray(times, dtype=float).ravel()
    censorship = np.asarray(censorship, dtype=float).ravel()
    risks = np.asarray(risks, dtype=float).ravel()

    valid_mask = ~(np.isnan(times) | np.isnan(censorship))
    times = times[valid_mask]
    censorship = censorship[valid_mask]
    risks = risks[valid_mask]

    n = len(times)
    if n == 0:
        return None
    n_events = int(np.sum(1.0 - censorship))
    if n_events == 0 or n < 2:
        return None

    events_observed = (1.0 - censorship).astype(bool)
    try:
        c_index = concordance_index_censored(events_observed, times, risks, tied_tol=1e-08)[0]
        return float(c_index)
    except Exception:
        return None


def compute_survival_metrics(hazards: np.ndarray, labels: np.ndarray):
    if labels is None or len(labels) == 0:
        return {}

    S = np.cumprod(1.0 - hazards, axis=1)
    risks = -np.sum(S, axis=1)

    survival_times = labels[:, 2]
    censorship = labels[:, 1]
    valid_mask = ~(np.isnan(survival_times) | np.isnan(censorship))
    if not valid_mask.any():
        return {}

    cidx = safe_concordance_index(survival_times[valid_mask], censorship[valid_mask], risks[valid_mask])
    if cidx is None:
        return {}

    return {
        'c_index': float(cidx),
        'n_samples': int(valid_mask.sum()),
        'mean_risk': float(np.mean(risks[valid_mask])),
        'std_risk': float(np.std(risks[valid_mask])),
    }


# -------------------------------
# Expert correlation matrix R + penalty
# -------------------------------
@torch.no_grad()
def compute_R_from_oof(oof_logits: torch.Tensor,
                       oof_labels: torch.Tensor,
                       shrink: float = 0.1,
                       pos_only: bool = True) -> torch.Tensor:
    """
    Estimate expert correlation matrix R from OOF hazard logits/labels.
    Uses per-expert survival NLL as the per-sample error signal.
    """
    oof_logits = oof_logits.to(dtype=torch.float32)
    device = oof_logits.device
    N, M, T = oof_logits.shape

    hazards = torch.sigmoid(oof_logits)  # [N, M, T]
    labels_t = oof_labels.to(device)

    losses = []
    for m in range(M):
        loss_m = survival_loss(hazards[:, m, :], labels_t, reduction='none')  # [N]
        losses.append(loss_m.detach())
    X = torch.stack(losses, dim=1)  # [N, M]

    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd

    R = (Z.t() @ Z) / max(Z.shape[0] - 1, 1)
    R = 0.5 * (R + R.t())

    if shrink > 0:
        diag_mean = torch.diag(R).mean()
        I = torch.eye(M, device=device)
        R = (1 - shrink) * R + shrink * diag_mean * I

    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(0.0)
    R_psd = (eigvecs * eigvals) @ eigvecs.t()

    if pos_only:
        R_psd = torch.clamp(R_psd, min=0.0)

    R_psd.fill_diagonal_(0.0)
    return R_psd


def func2(weights: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Quadratic penalty: E[w^T R w]
    Encourages diversity by penalizing correlated experts being weighted together.
    """
    assert weights.dim() == 2
    B, M = weights.shape
    assert R is not None and R.shape == (M, M)

    R_use = 0.5 * (R + R.t())
    R_use = torch.clamp(R_use, min=0.0)

    quad = torch.einsum("bm,mn,bn->b", weights, R_use, weights)  # [B]
    return quad.mean()


# -------------------------------
# Training loop
# -------------------------------
def train_with_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 600,
    lambda_l: float = DEFAULT_LAMBDA_L,
    R=None,
    lr: float = 1e-5,
):
    """
    Train gating network with survival loss + quadratic correlation penalty.
    Early stopping uses validation c-index.
    """
    training_start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    best_state = None
    best_val_cidx = -1.0
    patience, counter = 100, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_tot, train_surv_accum, train_pen_accum = 0.0, 0.0, 0.0

        for logits, labels in train_loader:
            logits = logits.to(device)
            labels = labels.to(device)

            x = survival_pred_feat_extraction(logits)
            fused_hazards, weights = model(x, logits)

            loss_surv = survival_loss(fused_hazards, labels)
            pen = func2(weights, R=R)
            loss = loss_surv + lambda_l * pen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_tot += loss.item()
            train_surv_accum += loss_surv.item()
            train_pen_accum += float(pen.detach().cpu())

        scheduler.step(epoch)

        # Validation (c-index)
        model.eval()
        hazards_list, labels_list = [], []
        val_pen_accum = 0.0
        with torch.no_grad():
            for logits, labels in val_loader:
                logits = logits.to(device)
                labels = labels.to(device)

                x = survival_pred_feat_extraction(logits)
                fused_hazards, weights = model(x, logits)

                hazards_list.append(fused_hazards.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                pen = func2(weights, R=R)
                val_pen_accum += float(pen.detach().cpu())

        H = np.concatenate(hazards_list, axis=0) if hazards_list else None
        Y = np.concatenate(labels_list, axis=0) if labels_list else None
        val_res = compute_survival_metrics(H, Y) if (H is not None and Y is not None) else {}
        val_cidx = float(val_res.get("c_index", 0.0))

        n_tr = max(1, len(train_loader))
        n_val = max(1, len(val_loader))
        avg_train_tot = train_tot / n_tr
        avg_train_surv = train_surv_accum / n_tr
        avg_train_pen = train_pen_accum / n_tr
        avg_val_pen = val_pen_accum / n_val

        if val_cidx > best_val_cidx:
            best_val_cidx = val_cidx
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        tqdm.write(
            f"[epoch {epoch:03d}] "
            f"train: total={avg_train_tot:.4f} (surv={avg_train_surv:.4f}, pen={avg_train_pen:.4f}) | "
            f"val: c_index={val_cidx:.4f}, pen={avg_val_pen:.4f} | "
            f"best_val_c_index={best_val_cidx:.4f}"
        )

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - training_start_time
    return model, training_time


# -------------------------------
# Evaluation helpers
# -------------------------------
@torch.no_grad()
def eval_on_split(model: nn.Module, logits: np.ndarray, labels: np.ndarray, device: torch.device):
    """
    Evaluate fused survival model on a split.
    Returns dict with c-index (if labels exist).
    """
    model.eval()
    N = logits.shape[0]
    batch = 64
    hazards_list = []

    for i in range(0, N, batch):
        lm = torch.from_numpy(logits[i:i + batch]).float().to(device)
        x = survival_pred_feat_extraction(lm)
        fused_hazards, _ = model(x, lm)
        hazards_list.append(fused_hazards.cpu().numpy())

    fused_hazards = np.concatenate(hazards_list, axis=0) if hazards_list else None
    results = {}
    if labels is not None and fused_hazards is not None and len(labels) == fused_hazards.shape[0]:
        results.update(compute_survival_metrics(fused_hazards, labels))
    return results


@torch.no_grad()
def print_model_weights(model: nn.Module, logits: np.ndarray, model_names: list, device: torch.device, title: str):
    """
    Print global mean expert weights across all samples in a split.
    """
    model.eval()
    B = logits.shape[0]
    batch = 64
    total_weights = None
    n_seen = 0

    for i in range(0, B, batch):
        lm = torch.from_numpy(logits[i:i + batch]).float().to(device)
        x = survival_pred_feat_extraction(lm)
        _, w = model(x, lm)  # [B, M]

        batch_sum = w.detach().cpu().sum(dim=0)  # [M]
        total_weights = batch_sum if total_weights is None else total_weights + batch_sum
        n_seen += w.shape[0]

    if n_seen == 0:
        print(f"{title}: no samples.")
        return None

    mean_weights = (total_weights / float(n_seen)).numpy()  # [M]

    print(f"\n===== {title}: Global Mean Expert Weights =====")
    order = np.argsort(mean_weights)[::-1]
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {model_names[idx]:<15} {mean_weights[idx]:.4f}")

    return mean_weights


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="POE hazard-logit fusion (GitHub release) - Survival")

    parser.add_argument(
        "--dataset",
        type=str,
        default="kirc",
        choices=["brca", "crc", "blca", "kirc", "lusc", "gbmlgg"],
        help="Dataset preset name (used for default root path only)."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory that contains per-model subfolders. "
             "If not set, defaults to ./outputs/<dataset>."
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["virchow2", "gigapath", "uni_v2", "hoptimus1", "conch_v15", "phikon_v2"],
        help="List of model folder names under <root>."
    )

    parser.add_argument("--method", type=str, default="poe_survival_per_timebin", help="Method name saved in outputs.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=600)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    if args.root is None:
        args.root = os.path.join(".", "outputs", args.dataset)

    model_root_paths = [os.path.join(args.root, d) for d in args.models if os.path.exists(os.path.join(args.root, d))]
    if not model_root_paths:
        raise RuntimeError(f"No valid model directories found under root={args.root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] dataset={args.dataset} root={args.root}")
    print(f"[info] models={args.models}")
    print(f"[info] device={device}")
    print(f"[info] defaults: lambda_l={DEFAULT_LAMBDA_L} (fixed), R=OOF (always ON)")

    all_val_results = []
    all_test_results = []

    for split_idx in range(args.n_splits):
        try:
            (train_s, val_s, test_s,
             model_names, n_models, n_bins) = load_multi_model_preds_and_labels(
                model_root_paths, split_idx
            )

            # unpack
            train_logits, train_ids, train_labels = train_s
            val_logits, val_ids, val_labels = val_s
            test_logits, test_ids, test_labels = test_s

            # Build OOF for R estimation (ALWAYS use other folds)
            if args.n_splits == 1:
                oof_logits = train_logits
                oof_labels = train_labels
            else:
                oof_logits, oof_labels, ref_names, ref_M, ref_T = build_oof_from_other_folds(
                    model_root_paths, cur_split=split_idx, n_splits=args.n_splits
                )
                assert ref_names == model_names, "OOF model order mismatch."
                assert ref_M == n_models and ref_T == n_bins, "OOF meta mismatch."

            R = compute_R_from_oof(
                torch.from_numpy(oof_logits).to(device),
                torch.from_numpy(oof_labels).float().to(device),
                shrink=DEFAULT_SHRINK,
                pos_only=DEFAULT_POS_ONLY,
            )

            # Dataset ONLY uses logits
            train_ds = SurvivalLogitDataset(train_logits, labels=train_labels)
            val_ds = SurvivalLogitDataset(val_logits, labels=val_labels)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

            model = POESurvivalPerTimeBin(
                n_models=n_models,
                in_dim=n_models * 3 + 2,
                n_bins=n_bins + 1,  # event bins + tail
                hidden=0,
                gate_temp=DEFAULT_GATE_TEMP,
                init_uniform=True,
                bias=True,
            ).to(device)

            model, training_time = train_with_val(
                model,
                train_loader,
                val_loader,
                device,
                num_epochs=args.num_epochs,
                lr=args.lr,
                R=R,
                lambda_l=DEFAULT_LAMBDA_L,  # fixed default
            )

            # Print weights (aligned with classification script behavior)
            print_model_weights(model, test_logits, model_names, device, title=f"Split {split_idx} Weights (Test)")

            # Evaluate fusion model
            val_res = eval_on_split(model, val_logits, val_labels, device)
            val_res.update({
                "split": split_idx,
                "method": args.method,
                "training_time_seconds": float(training_time),
            })
            all_val_results.append(val_res)

            test_res = eval_on_split(model, test_logits, test_labels, device)
            test_res.update({
                "split": split_idx,
                "method": args.method,
                "training_time_seconds": float(training_time),
            })
            all_test_results.append(test_res)

            if "c_index" in val_res:
                print(f"[split {split_idx}] VAL C-Index={val_res['c_index']:.4f} (n={val_res.get('n_samples', 0)})")
            else:
                print(f"[split {split_idx}] VAL (no labels)")

            if "c_index" in test_res:
                print(f"[split {split_idx}] TEST C-Index={test_res['c_index']:.4f} (n={test_res.get('n_samples', 0)})\n")
            else:
                print(f"[split {split_idx}] TEST (no labels)\n")

        except Exception as e:
            print(f"[split {split_idx}] failed: {e}")
            continue

    # -------------------------------
    # Summarize and save results
    # -------------------------------
    def summarize(results):
        if not results:
            return None

        cidxs = [r["c_index"] for r in results if "c_index" in r]
        ns = [r.get("n_samples", 0) for r in results if "n_samples" in r]
        times = [r.get("training_time_seconds", None) for r in results if "training_time_seconds" in r]
        times = [t for t in times if t is not None]

        return {
            "method": args.method,
            "splits": len(results),
            "mean_c_index": float(np.mean(cidxs)) if cidxs else None,
            "std_c_index": float(np.std(cidxs)) if cidxs else None,
            "mean_n_samples": float(np.mean(ns)) if ns else None,
            "mean_training_time_seconds": float(np.mean(times)) if times else None,
            "defaults": {
                "lambda_l": DEFAULT_LAMBDA_L,
                "R_from_oof": True,
                "shrink": DEFAULT_SHRINK,
                "pos_only": DEFAULT_POS_ONLY,
                "gate_temp": DEFAULT_GATE_TEMP,
            }
        }

    out_dir = os.path.join(args.root, "logits_fusion_results")
    os.makedirs(out_dir, exist_ok=True)

    val_summary = summarize(all_val_results)
    test_summary = summarize(all_test_results)

    if all_val_results:
        val_per_split = {
            r["split"]: {"val_c_index": r.get("c_index"), "val_n": r.get("n_samples")}
            for r in all_val_results if "split" in r
        }
        with open(os.path.join(out_dir, f"{args.method}_val_per_split.json"), "w") as f:
            json.dump(val_per_split, f, indent=2, ensure_ascii=False)

    if all_test_results:
        test_per_split = {
            r["split"]: {"test_c_index": r.get("c_index"), "test_n": r.get("n_samples")}
            for r in all_test_results if "split" in r
        }
        with open(os.path.join(out_dir, f"{args.method}_test_per_split.json"), "w") as f:
            json.dump(test_per_split, f, indent=2, ensure_ascii=False)

    if val_summary:
        with open(os.path.join(out_dir, f"{args.method}_val_summary.json"), "w") as f:
            json.dump(val_summary, f, indent=2, ensure_ascii=False)
        print(f"\n[save] VAL summary -> {os.path.join(out_dir, f'{args.method}_val_summary.json')}")

    if test_summary:
        with open(os.path.join(out_dir, f"{args.method}_test_summary.json"), "w") as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        print(f"[save] TEST summary -> {os.path.join(out_dir, f'{args.method}_test_summary.json')}")


if __name__ == "__main__":
    main()
