# LogitProd Training and Inference Pipeline

This document describes the workflow for training and inference using the LogitProd framework for both whole slide image (WSI)-level MIL tasks and patch-level classification, including ABMIL-based model training/inference and downstream logit fusion.

![Fig1](/Users/lianqi/Documents/PathLogitFed/Fig1_m.png)

## Overview

The pipeline consists of three main steps:
- **Step 1**: Feature extraction using Trident
- **Step 2**: Model training and inference (WSI-level MIL, survival analysis, gene mutation, and patch-level classification)
- **Step 3**: LogitProd logit-fusion aggregation and analysis

## Prerequisites

### Installation

#### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n LogitProd python=3.10 -y

# Activate the environment
conda activate LogitProd

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```


---

## Step 1: Extract Patch-level Features using Trident

Before training/inference (Step 2) and logit-fusion aggregation (Step 3), you need to extract patch-level features from whole slide images (WSI) using [Trident](https://github.com/mahmoodlab/Trident).

Please refer to the [Trident GitHub repository](https://github.com/mahmoodlab/Trident) for installation and feature extraction instructions.

The output should be patch-level features in the following structure:
```
<trident_processed>/
  └── 20x_256px/
      └── features_{model_name}/  # e.g., features_uni_v2/
          └── {slide_id}.h5
```

### Required Data Structure

After Step 1, ensure you have:

1. **Patch-level features**: Extracted features from Trident (as shown above)

2. **Data splits**: CSV files containing train/val/test splits
   ```
   <splits_dir>/
     └── splits_{split_idx}_k.csv  # e.g., splits_0_k.csv, splits_1_k.csv, ...
   ```

---

## Step 2: Model Training and Inference

In Step 2, you run **task-specific training and inference scripts** for four types of tasks.
All scripts are located under `scripts`:

- `WSI_classification/`
- `Gene_mutation/`
- `Survival_analysis/`
- `Patch_classification/`

Each folder contains its own **training** and **inference** scripts with consistent
GitHub-release style CLI arguments (paths parameterized, no hardcoded user paths).

### 2.1 WSI-level classification (ABMIL)

- **Scripts location**: `scripts/WSI_classification/`
- **Typical usage**:

```bash
cd scripts/WSI_classification

# Training
python train_abmil_WSI_classification.py --help

# Inference
python infer_abmil_WSI_classification.py --help
```

### 2.2 Gene mutation prediction

- **Scripts location**: `scripts/Gene_mutation/`
- **Typical usage**:

```bash
cd scripts/Gene_mutation

# Training
python train_abmil_Gene_mutation.py --help

# Inference
python infer_abmil_Gene_mutation.py --help
```

### 2.3 Survival analysis

- **Scripts location**: `scripts/Survival_analysis/`
- **Typical usage**:

```bash
cd scripts/Survival_analysis

# Training
python train_abmil_Survival_analysis.py --help

# Inference
python infer_abmil_Survival_analysis.py --help
```

### 2.4 Patch-level classification

- **Scripts location**: `scripts/Patch_classification/`
- **Typical usage**:

```bash
cd scripts/Patch_classification

# Training + inference are implemented in a single script
python train_infer_Patch_classification.py --help
```

---

## Step 3: LogitProd Aggregation

In Step 3, you run the **LogitProd** scripts to aggregate logits/features from
Step 2 via centralized logit fusion across tasks/models.

All LogitProd-related scripts live alongside the task scripts in:

- `scripts/WSI_classification/LogitProd_WSI_classification.py`
- `scripts/Gene_mutation/LogitProd_Gene_mutation.py`
- `scripts/Survival_analysis/LogitProd_Survival_analysis.py`
- `scripts/Patch_classification/LogitProd_Patch_classification.py`

### Typical usage

```bash
cd scripts/WSI_classification

python LogitProd_WSI_classification.py --help
```

You can choose the appropriate LogitProd script for:

- **WSI-level classification**: multi-model fusion of slide-level logits
- **Gene mutation prediction**: fusion of mutation logits from multiple expert models
- **Survival analysis**: fusion of survival-related logits/outputs from multiple expert models
- **Patch-level classification**: fusion of patch-level logits

Each script exposes task-specific CLI arguments (paths to logits / features from Step 2,
output directory for fusion results, etc.). Use `--help` to inspect the exact options.

---
