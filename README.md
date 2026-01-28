# PathLogitFed Training and Inference Pipeline

This document describes the workflow for training and inference using the PathLogitFed framework for both whole slide image (WSI)–level MIL tasks and patch-level classification, including federated training of ABMIL-based models and their inference.

## Overview

The pipeline consists of three main steps:
- **Step 1**: Feature extraction using Trident
- **Step 2**: Model training and inference (WSI-level MIL, survival analysis, gene mutation, and patch-level classification)
- **Step 3**: PathLogitFed federated aggregation and analysis

## Prerequisites

### Installation

#### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n PathLogitFed python=3.10 -y

# Activate the environment
conda activate PathLogitFed

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```


---

## Step 1: Extract Patch-level Features using Trident

Before training/inference (Step 2) and federated aggregation (Step 3), you need to extract patch-level features from whole slide images (WSI) using [Trident](https://github.com/mahmoodlab/Trident).

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
All scripts are located under `tutorials/code/scripts`:

- `WSI_classification/`
- `Gene_mutation/`
- `Survival_analysis/`
- `Patch_classification/`

Each folder contains its own **training** and **inference** scripts with consistent
GitHub-release style CLI arguments (paths parameterized, no hardcoded user paths).

### 2.1 WSI-level classification (ABMIL)

- **Scripts location**: `code/scripts/WSI_classification/`
- **Typical usage**:

```bash
cd tutorials/code/scripts/WSI_classification

# Training
python train_abmil_WSI_classification.py --help

# Inference
python infer_abmil_WSI_classification.py --help
```

### 2.2 Gene mutation prediction

- **Scripts location**: `code/scripts/Gene_mutation/`
- **Typical usage**:

```bash
cd tutorials/code/scripts/Gene_mutation

# Training
python train_abmil_Gene_mutation.py --help

# Inference
python infer_abmil_Gene_mutation.py --help
```

### 2.3 Survival analysis

- **Scripts location**: `code/scripts/Survival_analysis/`
- **Typical usage**:

```bash
cd tutorials/code/scripts/Survival_analysis

# Training
python train_abmil_Survival_analysis.py --help

# Inference
python infer_abmil_Survival_analysis.py --help
```

### 2.4 Patch-level classification

- **Scripts location**: `code/scripts/Patch_classification/`
- **Typical usage**:

```bash
cd tutorials/code/scripts/Patch_classification

# Training + inference are implemented in a single script
python train_infer_Patch_classification.py --help
```

---

## Step 3: PathLogitFed Aggregation

In Step 3, you run the **PathLogitFed** scripts to aggregate logits/features from
Step 2 across multiple cohorts / tasks in a federated manner.

All PathLogitFed-related scripts live alongside the task scripts in:

- `code/scripts/WSI_classification/logitsfed_WSI_classification.py`
- `code/scripts/Gene_mutation/logitsfed_Gene_mutation.py`
- `code/scripts/Survival_analysis/logitsfed_Survival_analysis.py`
- `code/scripts/Patch_classification/logitsfed_Patch_classification.py`

### Typical usage

```bash
cd tutorials/code/scripts/WSI_classification

python logitsfed_WSI_classification.py --help
```

You can choose the appropriate PathLogitFed script for:

- **WSI-level classification**: cross-cohort aggregation of slide-level logits
- **Gene mutation prediction**: aggregation across hospitals / cohorts for mutation tasks
- **Survival analysis**: aggregation of survival-related logits/outputs across cohorts (task-specific)
- **Patch-level classification**: aggregation of patch-level logits for federated analysis

Each script exposes task-specific CLI arguments (paths to logits / features from Step 2,
output directory for federated results, etc.). Use `--help` to inspect the exact options.

---



