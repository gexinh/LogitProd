#!/usr/bin/env python3
"""
POE Logit-based Fusion (Train/Val/Test) - GitHub Release Version

This script trains a lightweight gating network to fuse multiple expert models'
logits on slide-level classification tasks.

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
  python train_poe_logit_fusion.py --dataset crc --root ./outputs/crc
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

    def forward(self, x: torch.Tensor, logits: torch.Tensor):
        raw = self.gate(x) / max(self.gate_temp, 1e-8)   # [B, M]
        weights = F.softmax(raw, dim=-1)                 # [B, M]
        log_p = F.log_softmax(logits, dim=-1)            # [B, M, K]
        fused_logits = torch.einsum('bm,bmk->bk', weights, log_p)  # [B, K] (log-prob like)
        return fused_logits, weights


# -------------------------------
# Dataset (ONLY logits + labels)
# -------------------------------
class SlideLogitDataset(Dataset):
    """
    Minimal dataset for fusion training.
    We only need:
      - experts' logits: [N, M, C]
      - labels: [N]
    """
    def __init__(self, logits: np.ndarray, labels: np.ndarray = None):
        self.logits = torch.from_numpy(logits).float()
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


def build_oof_train_from_other_folds(model_root_paths, cur_split, n_splits):
    """
    Construct Out-of-Fold (OOF) data for estimating correlation matrix R.
    For each split != cur_split, collect its validation logits/labels.
    """
    oof_logits, oof_labels = [], []
    ref_model_names = None
    ref_meta = None  # (M, C, D)

    for s in range(n_splits):
        if s == cur_split:
            continue

        (train_s, val_s, test_s,
         model_names_s, n_models_s, n_classes_s, feat_dim_s) = load_multi_model_features_and_probs(model_root_paths, s)

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

        oof_logits.append(val_logits_s)
        oof_labels.append(val_labels_s)

    if not oof_logits:
        raise RuntimeError("No OOF data collected from other folds.")

    oof_logits = np.concatenate(oof_logits, axis=0)
    oof_labels = np.concatenate(oof_labels, axis=0)
    return oof_logits, oof_labels, ref_model_names, *ref_meta


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

    kind:
      - "nll": per-model negative log likelihood
      - "brier": per-model Brier score
      - "err01": per-model 0/1 error
    """
    oof_logits = oof_logits.to(dtype=torch.float32)
    device = oof_logits.device
    N, M, C = oof_logits.shape

    if kind == "nll":
        p = F.softmax(oof_logits, dim=-1)
        y = oof_labels.to(torch.long).view(-1, 1, 1).expand(-1, M, 1)
        X = -torch.log(p.gather(-1, y).clamp_min(eps)).squeeze(-1)  # [N, M]
    elif kind == "brier":
        p = F.softmax(oof_logits, dim=-1)
        onehot = torch.zeros_like(p).scatter_(-1, oof_labels.view(-1, 1, 1), 1.0)
        X = (p - onehot).pow(2).sum(dim=-1)  # [N, M]
    elif kind == "err01":
        pred = oof_logits.argmax(dim=-1)
        X = (pred != oof_labels.view(-1, 1)).float()
    else:
        raise ValueError("kind must be one of: 'nll' | 'brier' | 'err01'")

    # Standardize each model dimension
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd  # [N, M]

    # Correlation estimate
    R = (Z.t() @ Z) / (Z.shape[0] - 1)
    R = 0.5 * (R + R.t())

    # Shrinkage stabilization
    if shrink > 0:
        diag_mean = torch.diag(R).mean()
        I = torch.eye(M, device=device)
        R = (1 - shrink) * R + shrink * diag_mean * I

    # PSD projection
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(0.0)
    R_psd = (eigvecs * eigvals) @ eigvecs.t()

    if pos_only:
        R_psd = torch.clamp(R_psd, min=0.0)

    # Remove diagonal self-correlation for penalty
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
    num_epochs: int = 30,
    lambda_l: float = 1e-1,
    R=None,
    lr: float = 5e-4,
):
    """
    Train gating network with CE loss + quadratic correlation penalty.
    Early stopping uses validation CE only.
    """
    training_start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    ce = nn.CrossEntropyLoss()

    best_state = None
    best_val_ce = float("inf")
    patience, counter = 100, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_tot, train_ce_accum, train_pen_accum = 0.0, 0.0, 0.0

        for logits, labels in train_loader:
            logits = logits.to(device)
            labels = labels.to(device)

            x = logit_feat_extraction(logits)
            fused_logits, weights = model(x, logits)

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
        avg_train_tot = train_tot / n_tr
        avg_train_ce = train_ce_accum / n_tr
        avg_train_pen = train_pen_accum / n_tr

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
                pen = func2(weights, R=R)

                val_ce_accum += loss_ce.item()
                val_pen_accum += float(pen.detach().cpu())

        n_val = max(1, len(val_loader))
        avg_val_ce = val_ce_accum / n_val
        avg_val_pen = val_pen_accum / n_val

        if avg_val_ce < best_val_ce:
            best_val_ce = avg_val_ce
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        tqdm.write(
            f"[epoch {epoch:03d}] "
            f"train: total={avg_train_tot:.4f} (ce={avg_train_ce:.4f}, pen={avg_train_pen:.4f}) | "
            f"val: ce={avg_val_ce:.4f}, pen={avg_val_pen:.4f} | "
            f"best_val_ce={best_val_ce:.4f}"
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
    Evaluate fused classifier on a split.
    Returns dict with auc/acc/f1 (if labels exist).
    """
    model.eval()
    B = logits.shape[0]
    batch = 64
    outs = []

    for i in range(0, B, batch):
        lm = torch.from_numpy(logits[i:i + batch]).float().to(device)
        x = logit_feat_extraction(lm)
        fused_logits, _ = model(x, lm)
        outs.append(fused_logits.cpu().numpy())

    fused_logits = np.concatenate(outs, axis=0)
    fused_probs = torch.softmax(torch.tensor(fused_logits), dim=1).numpy()
    preds = np.argmax(fused_logits, axis=1)

    results = {}
    if labels is not None and len(labels) == fused_probs.shape[0]:
        try:
            if fused_probs.shape[1] == 2:
                auc = roc_auc_score(labels, fused_probs[:, 1])
            else:
                auc = roc_auc_score(labels, fused_probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.5

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds) if fused_probs.shape[1] == 2 else f1_score(labels, preds, average="macro")
        results.update({"auc": float(auc), "acc": float(acc), "f1": float(f1)})

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
        x = logit_feat_extraction(lm)
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
    parser = argparse.ArgumentParser(description="POE logit-based fusion (GitHub release)")

    parser.add_argument(
        "--dataset",
        type=str,
        default="crc",
        choices=["brca", "bracs", "bracs7", "crc", "panda"],
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
        default=["virchow2", "uni_v2", "lunit-vits8", "gigapath", "hoptimus1", "conch_v15", "ctranspath"],
        help="List of model folder names under <root>."
    )

    parser.add_argument("--method", type=str, default="poe", help="Method name saved in outputs.")
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

    # Optional: read individual model metrics from summary.json
    def read_individual_model_results(model_root_paths, model_names, split_idx):
        val_results, test_results = {}, {}
        for model_path, model_name in zip(model_root_paths, model_names):
            summary_file = os.path.join(model_path, "summary.json")
            if not os.path.exists(summary_file):
                continue

            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)

                individual_results = summary.get("individual_results", [])
                for result in individual_results:
                    if result.get("split_idx") == split_idx:
                        val_results[model_name] = {
                            "auc": result.get("best_val_auc", None),
                            "acc": result.get("best_val_acc", None),
                            "f1": result.get("best_val_f1", None),
                        }
                        test_results[model_name] = {
                            "auc": result.get("test_auc", None),
                            "acc": result.get("test_acc", None),
                            "f1": result.get("test_f1", None),
                        }
                        break
            except Exception as e:
                print(f"[warn] Failed reading {summary_file}: {e}")

        return val_results, test_results

    all_val_results = []
    all_test_results = []

    for split_idx in range(args.n_splits):
        try:
            (train_s, val_s, test_s, model_names, n_models, n_classes, feat_dim) = load_multi_model_features_and_probs(
                model_root_paths, split_idx
            )

            # unpack
            train_feats, train_logits, train_ids, train_labels = train_s
            val_feats, val_logits, val_ids, val_labels = val_s
            test_feats, test_logits, test_ids, test_labels = test_s

            # Build OOF for R estimation
            if args.n_splits == 1:
                oof_logits = train_logits
                oof_labels = train_labels
            else:
                oof_logits, oof_labels, ref_names, ref_M, ref_C, ref_D = build_oof_train_from_other_folds(
                    model_root_paths, cur_split=split_idx, n_splits=args.n_splits
                )
                assert ref_names == model_names, "OOF model order mismatch."
                assert ref_M == n_models and ref_C == n_classes and ref_D == feat_dim, "OOF meta mismatch."

            R = compute_R_from_oof(
                torch.from_numpy(oof_logits).to(device),
                torch.from_numpy(oof_labels).long().to(device),
                kind="nll",
                shrink=0.1,
            )

            # Dataset ONLY uses logits (features removed)
            train_ds = SlideLogitDataset(train_logits, labels=train_labels)
            val_ds = SlideLogitDataset(val_logits, labels=val_labels)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

            model = POE(
                n_models=n_models,
                in_dim=n_models * 3 + 2,
                hidden=0,
            ).to(device)

            model, training_time = train_with_val(
                model,
                train_loader,
                val_loader,
                device,
                num_epochs=args.num_epochs,
                lr=args.lr,
                R=R,
            )

            # Optional: print individual model metrics if available
            individual_val_results, individual_test_results = read_individual_model_results(
                model_root_paths, model_names, split_idx
            )

            if individual_val_results:
                print(f"\n===== Split {split_idx} Individual Model Results (Val) =====")
                for model_name, results in individual_val_results.items():
                    auc_str = f"{results['auc']:.4f}" if results["auc"] is not None else "N/A"
                    acc_str = f"{results['acc']:.4f}" if results["acc"] is not None else "N/A"
                    f1_str = f"{results['f1']:.4f}" if results["f1"] is not None else "N/A"
                    print(f"  {model_name:15s}: AUC={auc_str:>8s} ACC={acc_str:>8s} F1={f1_str:>8s}")

            print_model_weights(model, test_logits, model_names, device, title=f"Split {split_idx} Weights (Test)")

            if individual_test_results:
                print(f"\n===== Split {split_idx} Individual Model Results (Test) =====")
                for model_name, results in individual_test_results.items():
                    auc_str = f"{results['auc']:.4f}" if results["auc"] is not None else "N/A"
                    acc_str = f"{results['acc']:.4f}" if results["acc"] is not None else "N/A"
                    f1_str = f"{results['f1']:.4f}" if results["f1"] is not None else "N/A"
                    print(f"  {model_name:15s}: AUC={auc_str:>8s} ACC={acc_str:>8s} F1={f1_str:>8s}")

            # Evaluate fusion model
            val_res = eval_on_split(model, val_logits, val_labels, device)
            val_res.update({
                "split": split_idx,
                "method": args.method,
                "individual_models": individual_val_results,
                "training_time_seconds": float(training_time),
            })
            all_val_results.append(val_res)

            test_res = eval_on_split(model, test_logits, test_labels, device)
            test_res.update({
                "split": split_idx,
                "method": args.method,
                "individual_models": individual_test_results,
                "training_time_seconds": float(training_time),
            })
            all_test_results.append(test_res)

            if "auc" in val_res:
                print(f"[split {split_idx}] VAL AUC={val_res['auc']:.4f} ACC={val_res['acc']:.4f} F1={val_res['f1']:.4f}")
            else:
                print(f"[split {split_idx}] VAL (no labels)")

            if "auc" in test_res:
                print(f"[split {split_idx}] TEST AUC={test_res['auc']:.4f} ACC={test_res['acc']:.4f} F1={test_res['f1']:.4f}")
            else:
                print(f"[split {split_idx}] TEST (no labels)")

        except Exception as e:
            print(f"[split {split_idx}] failed: {e}")
            continue

    # -------------------------------
    # Summarize and save results
    # -------------------------------
    def summarize(results):
        if not results:
            return None

        aucs = [r["auc"] for r in results if "auc" in r]
        accs = [r["acc"] for r in results if "acc" in r]
        f1s = [r["f1"] for r in results if "f1" in r]
        times = [r.get("training_time_seconds", None) for r in results if "training_time_seconds" in r]
        times = [t for t in times if t is not None]

        return {
            "method": args.method,
            "splits": len(results),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
            "mean_acc": float(np.mean(accs)) if accs else None,
            "std_acc": float(np.std(accs)) if accs else None,
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "std_f1": float(np.std(f1s)) if f1s else None,
            "mean_training_time_seconds": float(np.mean(times)) if times else None,
        }

    out_dir = os.path.join(args.root, "logits_fusion_results")
    os.makedirs(out_dir, exist_ok=True)

    val_summary = summarize(all_val_results)
    test_summary = summarize(all_test_results)

    if all_val_results:
        val_per_split = {
            r["split"]: {"val_auc": r.get("auc"), "val_acc": r.get("acc"), "val_f1": r.get("f1")}
            for r in all_val_results if "split" in r
        }
        with open(os.path.join(out_dir, f"{args.method}_val_per_split.json"), "w") as f:
            json.dump(val_per_split, f, indent=2, ensure_ascii=False)

    if all_test_results:
        test_per_split = {
            r["split"]: {"test_auc": r.get("auc"), "test_acc": r.get("acc"), "test_f1": r.get("f1")}
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

    # Per-model aggregated stats (test)
    print("\n===== Individual Model Performance Summary (Test) =====")
    individual_model_stats = {}
    for result in all_test_results:
        if "individual_models" not in result:
            continue
        for model_name, metrics in result["individual_models"].items():
            if model_name not in individual_model_stats:
                individual_model_stats[model_name] = {"auc": [], "acc": [], "f1": []}
            for metric in ["auc", "acc", "f1"]:
                if metrics.get(metric) is not None:
                    individual_model_stats[model_name][metric].append(metrics[metric])

    if individual_model_stats:
        for model_name in sorted(individual_model_stats.keys()):
            stats = individual_model_stats[model_name]
            auc_mean = np.mean(stats["auc"]) if stats["auc"] else 0
            auc_std = np.std(stats["auc"]) if stats["auc"] else 0
            acc_mean = np.mean(stats["acc"]) if stats["acc"] else 0
            acc_std = np.std(stats["acc"]) if stats["acc"] else 0
            f1_mean = np.mean(stats["f1"]) if stats["f1"] else 0
            f1_std = np.std(stats["f1"]) if stats["f1"] else 0
            print(f"  {model_name:15s}: AUC={auc_mean:.4f}±{auc_std:.4f} "
                  f"ACC={acc_mean:.4f}±{acc_std:.4f} F1={f1_mean:.4f}±{f1_std:.4f}")
    else:
        print("No individual model summaries found (summary.json missing).")


if __name__ == "__main__":
    main()