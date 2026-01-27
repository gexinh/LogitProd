#!/usr/bin/env python3
"""
ABMIL Model Inference for Gene Mutation Classification - GitHub Release Version

This script performs inference using saved best_model.pth checkpoints for each cross-validation split.
It re-runs inference on train/val/test sets and saves results with optional date suffix.

Key characteristics:
  - Loads best_model.pth from training output directory for each split.
  - Performs inference on train/val/test sets with the saved model.
  - Extracts and saves slide-level features for downstream fusion tasks.
  - Supports gene mutation binary classification with cross-validation splits.
  - Saves results with optional date suffix for versioning.

Supported datasets:
  - brca_lusc: BRCA+LUSC gene mutation binary classification

Expected directory structure:
  <train_output_root>/<model_name>/split_<k>/
    - best_model.pth (saved model checkpoint)
  <feats_path>/
    - features_{model_name}/ (patch-level features)
  <splits_dir>/
    - splits_{split_idx}_k.csv (train/val/test splits)
  <mutation_labels_file>/
    - CSV file containing gene mutation labels

Output directory structure:
  <infer_output_root>/<model_name>/split_<k>/
    - metrics{date_suffix}.json (inference metrics)
    - best_train_outputs.npy, best_val_outputs.npy, test_outputs.npy
    - best_train_labels.npy, best_val_labels.npy, test_labels.npy
    - best_train_predictions.npy, best_val_predictions.npy, test_predictions.npy
    - best_train_probabilities.npy, best_val_probabilities.npy, test_probabilities.npy
    - train_features_labels.h5, val_features_labels.h5, test_features_labels.h5
  <infer_output_root>/<model_name>/
    - summary.json (aggregated results across splits)

Usage example:
  python infer_abmil_genemutation.py \
      --target_gene TP53 \
      --feats_path /path/to/features_{MODEL_NAME} \
      --splits_dir /path/to/splits \
      --train_output_root /path/to/train_outputs \
      --infer_output_root /path/to/infer_outputs/{target_gene} \
      --mutation_labels_file /path/to/labels.csv
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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
    from brca_lusc_dataset import create_brca_lusc_classification_dataloaders
    BRCA_LUSC_AVAILABLE = True
except ImportError:
    BRCA_LUSC_AVAILABLE = False
    print("Warning: brca_lusc_dataset not available")

# -------------------------------
# Model import
# -------------------------------
# Get the models directory (two levels up from scripts subfolder)
model_dir = os.path.abspath(os.path.join(script_dir, '../../models'))
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
from model_abmil_with_feat import AttentionGatedWithFeat

# -------------------------------
# Model hyperparameters
# -------------------------------
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
# Data utilities
# -------------------------------
def custom_collate_fn(batch):
    """
    Custom collate function for handling variable-length feature sequences.
    Handles dict labels from brca_lusc dataset.
    """
    features_list = []
    labels = []
    for features, label in batch:
        if features is None or label is None:
            continue
        features_list.append(features)
        # Handle dict labels from brca_lusc dataset
        if isinstance(label, dict):
            # Extract numeric label from dict
            if 'label' in label:
                # brca_lusc dataset returns label as torch.tensor
                if isinstance(label['label'], torch.Tensor):
                    labels.append(label['label'].item())
                else:
                    labels.append(label['label'])
            else:
                # Fallback: try to extract any numeric value
                labels.append(list(label.values())[0] if label else 0)
        else:
            labels.append(label)
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
    def __init__(self, input_feature_dim=1024, n_classes=2, act='relu', bias=False, dropout=False):
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

# -------------------------------
# Evaluation utilities
# -------------------------------
def compute_metrics_from_logits(all_outputs, all_labels):
    """
    Compute metrics (AUC, accuracy, F1) from logits.
    Handles binary classification for gene mutation.
    """
    preds = np.argmax(all_outputs, axis=1)
    probabilities = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
    # AUC for binary classification
    auc = roc_auc_score(all_labels, probabilities[:, 1])
    f1 = f1_score(all_labels, preds)
    accuracy = accuracy_score(all_labels, preds)
    return auc, accuracy, f1, probabilities, preds

# -------------------------------
# Inference
# -------------------------------
@torch.no_grad()
def infer_loader(model, data_loader, device, desc):
    """
    Perform inference on a data loader.
    Returns metrics (AUC, accuracy, F1), outputs, and labels.
    """
    model.eval()
    all_labels = []
    all_outputs = []
    pbar = tqdm(data_loader, desc=desc, leave=False)
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
@torch.no_grad()
def extract_features(model, data_loader, device, set_name):
    """
    Extract slide-level features from model.
    Returns features, labels, and outputs for saving.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_outputs = []
    pbar = tqdm(data_loader, desc=f"Extract {set_name}", leave=False)
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
        return None, None, None
    all_outputs = np.concatenate(all_outputs)
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels, all_outputs

def save_features_to_h5(features, labels, outputs, save_path):
    """Save slide-level features, labels, and outputs to HDF5 file."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('labels', data=labels)
        f.create_dataset('outputs', data=outputs)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Re-infer ABMIL best models for gene mutation classification")
    parser.add_argument('--target_gene', type=str, required=True, choices=['TP53', 'PIK3CA', 'PTEN', 'KRAS', 'ARID1A', 'NF1', 'BRCA2'],
                        help='Target gene for binary classification')
    parser.add_argument('--feats_path', type=str, required=True,
                        help='Path to patch-level features directory. Supports {MODEL_NAME} placeholder.')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Path to splits directory containing CSV files.')
    parser.add_argument('--train_output_root', type=str, required=True,
                        help='Root directory of training outputs (contains best_model.pth for each split).')
    parser.add_argument('--infer_output_root', type=str, required=True,
                        help='Root directory for inference outputs. Supports {target_gene} placeholder.')
    parser.add_argument('--mutation_labels_file', type=str, required=True,
                        help='Path to mutation labels CSV file.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--date_suffix', type=str, default='', help='Optional suffix for output file names (e.g., _1112)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use provided arguments (all paths are required)
    # Support {MODEL_NAME} placeholder in feats_path
    if '{MODEL_NAME}' in args.feats_path:
        feats_root_tmpl = args.feats_path
    else:
        # If no placeholder, append features_{MODEL_NAME} to the path
        feats_root_tmpl = os.path.join(args.feats_path, "features_{model}")

    splits_dir = args.splits_dir
    train_output_root = args.train_output_root

    # Support {target_gene} placeholder in infer_output_root
    if '{target_gene}' in args.infer_output_root:
        infer_output_root = args.infer_output_root.format(target_gene=args.target_gene)
    else:
        infer_output_root = args.infer_output_root

    # Binary classification for gene mutation
    n_classes = 2
    target_gene = args.target_gene
    mutation_labels_file = args.mutation_labels_file

    num_splits = args.num_splits

    for MODEL_NAME, model_params in MODEL_HYPERPARAMS.items():
        input_feature_dim = model_params["dim"]
        feats_path = feats_root_tmpl.format(model=MODEL_NAME)
        if not os.path.exists(feats_path) or not os.path.exists(splits_dir):
            print(f"[skip] Missing path for model {MODEL_NAME}: feats={feats_path} or splits={splits_dir}")
            continue

        output_base_dir_train = os.path.join(train_output_root, MODEL_NAME)
        output_base_dir_infer = os.path.join(infer_output_root, MODEL_NAME)
        os.makedirs(output_base_dir_infer, exist_ok=True)
        if not os.path.exists(output_base_dir_train):
            print(f"[skip] Train output dir not found for model {MODEL_NAME}: {output_base_dir_train}")
            continue

        print(f"\n===== Re-infer model {MODEL_NAME} for {target_gene} gene mutation ({num_splits} folds) =====")
        all_results = []
        for split_idx in range(num_splits):
            split_output_dir_train = os.path.join(output_base_dir_train, f"split_{split_idx}")
            split_output_dir_infer = os.path.join(output_base_dir_infer, f"split_{split_idx}")
            os.makedirs(split_output_dir_infer, exist_ok=True)
            best_model_path = os.path.join(split_output_dir_train, "best_model.pth")
            if not os.path.exists(best_model_path):
                print(f"[skip] best_model.pth not found: {best_model_path}")
                continue

            # Build loaders for this split
            if not BRCA_LUSC_AVAILABLE:
                print("brca_lusc_dataset not available")
                continue
            
            train_loader, val_loader, test_loader = create_brca_lusc_classification_dataloaders(
                feats_path=feats_path,
                mutation_labels_file=mutation_labels_file,
                splits_dir=splits_dir,
                split_idx=split_idx,
                batch_size=args.batch_size,
                target_gene=target_gene
            )

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
            test_loader = torch.utils.data.DataLoader(
                test_loader.dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

            # Build and load model
            model = BinaryClassificationModel(
                input_feature_dim=input_feature_dim,
                n_classes=n_classes,
                act='relu',
                bias=False,
                dropout=True
            ).to(device)
            state_dict = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state_dict)

            # Inference
            train_auc, train_acc, train_f1, train_outputs, train_labels = infer_loader(model, train_loader, device, "Re-train")
            val_auc, val_acc, val_f1, val_outputs, val_labels = infer_loader(model, val_loader, device, "Re-val")
            test_auc, test_acc, test_f1, test_outputs, test_labels = infer_loader(model, test_loader, device, "Re-test")

            # Derive predictions and probabilities
            train_probs = torch.softmax(torch.from_numpy(train_outputs), dim=1).numpy() if train_outputs is not None else np.array([])
            val_probs = torch.softmax(torch.from_numpy(val_outputs), dim=1).numpy() if val_outputs is not None else np.array([])
            test_probs = torch.softmax(torch.from_numpy(test_outputs), dim=1).numpy() if test_outputs is not None else np.array([])
            train_preds = np.argmax(train_outputs, axis=1) if train_outputs is not None else np.array([])
            val_preds = np.argmax(val_outputs, axis=1) if val_outputs is not None else np.array([])
            test_preds = np.argmax(test_outputs, axis=1) if test_outputs is not None else np.array([])

            # Save NPY/JSON with suffix
            suffix = args.date_suffix if args.date_suffix else ""
            if suffix and not suffix.startswith('_'):
                suffix = f"_{suffix}"
            # metrics (large file: use suffix)
            results = {
                'model_name': MODEL_NAME,
                'split_idx': split_idx,
                'target_gene': target_gene,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                # unified metrics from re-infer (best val now equals current val metrics)
                'best_val_auc': float(val_auc) if val_auc is not None else None,
                'best_val_acc': float(val_acc) if val_acc is not None else None,
                'best_val_f1': float(val_f1) if val_f1 is not None else None,
                'train_auc': float(train_auc) if train_auc is not None else None,
                'train_acc': float(train_acc) if train_acc is not None else None,
                'train_f1': float(train_f1) if train_f1 is not None else None,
                'test_auc': float(test_auc) if test_auc is not None else None,
                'test_acc': float(test_acc) if test_acc is not None else None,
                'test_f1': float(test_f1) if test_f1 is not None else None,
            }
            with open(os.path.join(split_output_dir_infer, f'metrics{suffix}.json'), 'w') as f:
                json.dump(results, f, indent=2)

            # train (per-split small files: use original naming, no suffix)
            if train_outputs is not None and len(train_outputs) > 0:
                np.save(os.path.join(split_output_dir_infer, 'best_train_outputs.npy'), train_outputs)
                np.save(os.path.join(split_output_dir_infer, 'best_train_predictions.npy'), train_preds)
                np.save(os.path.join(split_output_dir_infer, 'best_train_labels.npy'), train_labels)
                np.save(os.path.join(split_output_dir_infer, 'best_train_probabilities.npy'), train_probs)
                np.save(os.path.join(split_output_dir_infer, 'best_train_results.npy'), {
                    'best_train_outputs': train_outputs,
                    'best_train_predictions': train_preds,
                    'best_train_labels': train_labels,
                    'best_train_probabilities': train_probs
                })
            # val (per-split small files: use original naming, no suffix)
            if val_outputs is not None and len(val_outputs) > 0:
                np.save(os.path.join(split_output_dir_infer, 'best_val_outputs.npy'), val_outputs)
                np.save(os.path.join(split_output_dir_infer, 'best_val_predictions.npy'), val_preds)
                np.save(os.path.join(split_output_dir_infer, 'best_val_labels.npy'), val_labels)
                np.save(os.path.join(split_output_dir_infer, 'best_val_probabilities.npy'), val_probs)
                np.save(os.path.join(split_output_dir_infer, 'best_val_results.npy'), {
                    'best_val_outputs': val_outputs,
                    'best_val_predictions': val_preds,
                    'best_val_labels': val_labels,
                    'best_val_probabilities': val_probs
                })
            # test (per-split small files: use original naming, no suffix)
            if test_outputs is not None and len(test_outputs) > 0:
                np.save(os.path.join(split_output_dir_infer, 'test_outputs.npy'), test_outputs)
                np.save(os.path.join(split_output_dir_infer, 'test_predictions.npy'), test_preds)
                np.save(os.path.join(split_output_dir_infer, 'test_labels.npy'), test_labels)
                np.save(os.path.join(split_output_dir_infer, 'test_probabilities.npy'), test_probs)
                np.save(os.path.join(split_output_dir_infer, 'test_results.npy'), {
                    'test_outputs': test_outputs,
                    'test_predictions': test_preds,
                    'test_labels': test_labels,
                    'test_probabilities': test_probs
                })

            # Re-extract and save features H5 (per-split small files: use original naming, no suffix)
            re_train_feats, re_train_labels, re_train_outs = extract_features(model, train_loader, device, "train")
            if re_train_feats is not None:
                save_features_to_h5(re_train_feats, re_train_labels, re_train_outs,
                                    os.path.join(split_output_dir_infer, 'train_features_labels.h5'))
            re_val_feats, re_val_labels, re_val_outs = extract_features(model, val_loader, device, "val")
            if re_val_feats is not None:
                save_features_to_h5(re_val_feats, re_val_labels, re_val_outs,
                                    os.path.join(split_output_dir_infer, 'val_features_labels.h5'))
            re_test_feats, re_test_labels, re_test_outs = extract_features(model, test_loader, device, "test")
            if re_test_feats is not None:
                save_features_to_h5(re_test_feats, re_test_labels, re_test_outs,
                                    os.path.join(split_output_dir_infer, 'test_features_labels.h5'))

            # Collect for summary
            all_results.append({
                'model_name': MODEL_NAME,
                'split_idx': split_idx,
                'train_auc': results['train_auc'],
                'train_acc': results['train_acc'],
                'train_f1': results['train_f1'],
                'best_val_auc': results['best_val_auc'],
                'best_val_acc': results['best_val_acc'],
                'best_val_f1': results['best_val_f1'],
                'test_auc': results['test_auc'],
                'test_acc': results['test_acc'],
                'test_f1': results['test_f1'],
            })

        # Write summary with unified best val metrics
        if all_results:
            def safe_mean(values):
                values = [v for v in values if v is not None]
                return float(np.mean(values)) if values else None
            def safe_std(values):
                values = [v for v in values if v is not None]
                return float(np.std(values)) if values else None
            summary = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'date_suffix': args.date_suffix,
                'target_gene': target_gene,
                'model_name': MODEL_NAME,
                'splits': len(all_results),
                'mean_train_auc': safe_mean([r['train_auc'] for r in all_results]),
                'std_train_auc': safe_std([r['train_auc'] for r in all_results]),
                'mean_train_acc': safe_mean([r['train_acc'] for r in all_results]),
                'std_train_acc': safe_std([r['train_acc'] for r in all_results]),
                'mean_train_f1': safe_mean([r['train_f1'] for r in all_results]),
                'std_train_f1': safe_std([r['train_f1'] for r in all_results]),
                'mean_best_val_auc': safe_mean([r['best_val_auc'] for r in all_results]),
                'std_best_val_auc': safe_std([r['best_val_auc'] for r in all_results]),
                'mean_best_val_acc': safe_mean([r['best_val_acc'] for r in all_results]),
                'std_best_val_acc': safe_std([r['best_val_acc'] for r in all_results]),
                'mean_best_val_f1': safe_mean([r['best_val_f1'] for r in all_results]),
                'std_best_val_f1': safe_std([r['best_val_f1'] for r in all_results]),
                'mean_test_auc': safe_mean([r['test_auc'] for r in all_results]),
                'std_test_auc': safe_std([r['test_auc'] for r in all_results]),
                'mean_test_acc': safe_mean([r['test_acc'] for r in all_results]),
                'std_test_acc': safe_std([r['test_acc'] for r in all_results]),
                'mean_test_f1': safe_mean([r['test_f1'] for r in all_results]),
                'std_test_f1': safe_std([r['test_f1'] for r in all_results]),
                'individual_results': all_results
            }
            with open(os.path.join(output_base_dir_infer, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {os.path.join(output_base_dir_infer, 'summary.json')}")
        else:
            print(f"No splits processed for model {MODEL_NAME}")

if __name__ == "__main__":
    main()
