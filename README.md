# 3Di VAE v2: Modernized 3Di Structural Alphabet Encoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

A modernized, config-driven PyTorch/Lightning pipeline for training and evaluating 3Di structural alphabet VQ-VAE models.

> [!NOTE]
> **Legacy v1 Implementation**: The original v1 codebase is located under `src/tdi/v1/`. All active development, configuration-driven features, training optimizations, and evaluation tools are centered on the modernized **v2** pipeline under `src/tdi/v2/`.

---

## Key Features in v2

- **Quantizer Backends**:
  - **EMAVectorQuantizer**: Exponential Moving Average codebook updates, including L2 normalization, k-means centroid seeding, and automated dead code replacement.
  - **FSQQuantizer**: Finite Scalar Quantizer (FSQ) baseline that removes codebook updates in favor of fixed grid quantization.
- **Hybrid Optimization Objective**:
  - **Reconstruction Loss**: Reconstructs both self-descriptors and aligned partner-descriptors (supporting `smooth_l1` or `gaussian_nll`).
  - **Contrastive Learning**: In-batch negative-negative representation matching with learnable temperature scaling.
  - **Auxiliary Losses**: Commitment loss and usage entropy regularizer to maximize codebook utilization.
- **Numerical & Gradient Stability**:
  - **Rotation Trick**: Householder reflection/rotation style gradient routing as the default for VQ-VAEs.
  - **Deterministic CIGAR Expansion**: SVD/Kabsch-based 3D superposition filtering for residue coordinate alignment.
  - **Contiguous Views & FP32 Autocast Bypasses**: High-precision matrix maths and distance computations to prevent precision underflow under mixed training.
- **Flexible Standalone Deployment**:
  - Model weights, scale factors, and centroids can be exported to lightweight modular artifacts (`.pt`, `.npy`, `.json`) for standalone inference or integration into C++ environments (e.g., Foldseek).

---

## Getting Started

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager

### Installation
Sync project dependencies inside the virtual environment:
```bash
uv sync
```

---

## Usage Workflow

### 1. Data Preprocessing
Generate standardized features, standardizer scales, and metadata parquets from structural alignments and PDB files:
```bash
uv run python -m tdi.data build-features --config configs/data/scop.yaml --force
```

This creates the processed dataset in `data/processed/scop_ca5_v1/`, including:
- `train_x_raw.npy` / `train_y_raw.npy` (Cast to `float32`)
- `scaler.npz` (mean/std normalization vectors)
- Skew/QC reports and metadata parquets

### 2. Training a Model
Train the v2 model using PyTorch Lightning and the YAML training configuration:
```bash
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml
```

*Note: You can override parameters on the command line using dotted notation (e.g., `--training.max_epochs 10` or `--outputs.out_dir outputs/my_model`).*

### 3. Evaluating a Model
Compute alphabet metrics (perplexity, mutual information, state entropy) and generate structural scoring matrices (`submat.txt`):
```bash
uv run python -m tdi.v2 evaluate \
  --model_dir outputs/models/scop_v2_default_seed1 \
  --pdb_dir data/pdb \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out_dir outputs/eval
```

---

## Verification & Testing

Execute the comprehensive test suite (79 tests) with `pytest`:
```bash
uv run pytest
```

---

## Directory Structure

```
├── configs/            # YAML configs for data generation and model training
├── data/
│   ├── raw/            # Baseline SCOPe SIDs and alignment pairfiles
│   ├── derived/        # Train/val split files
│   └── processed/      # Preprocessed float32 feature numpy arrays
├── docs/               # Detailed feature, training, and evaluation docs
├── src/tdi/
│   ├── data/           # Preprocessing pipeline orchestration
│   ├── v1/             # Legacy v1 codebase
│   └── v2/             # Modernized VAE, quantizers, training & evaluation
└── tests/              # Multi-tiered unit and integration tests
```
