# 3Di Structural Alphabet Documentation

This directory contains architectural and reference documentation for the 3Di VAE v2 project.

---

## Architecture Overview

The system is organized into two primary layers:

1. **Preprocessing Layer (`src/tdi/data/`)**
   - Parses PDB coordinates, extracts 10-D geometric descriptors, superposes aligned pairs via Kabsch/SVD filtering, fits train-only feature standardizers, and outputs immutable, content-hashed datasets with QC reports and data cards.
   - Command: `python -m tdi.data <build-features|validate|report> --config ...`

2. **Model & Alphabet Learning Layer (`src/tdi/v2/`)**
   - Core MLP encoder-decoder architecture with two first-class discrete quantizers: **EMA-VQ** (reference learner with dead-code replacement and k-means initialization) and **FSQ** (finite scalar comparator).
   - Trains using plain PyTorch (fp32 throughout, straight-through estimator) and produces self-describing export artifacts.
   - Evaluates trained models on held-out structural alignments to emit substitution matrices and alphabet mutual information metrics.
   - Commands: `python -m tdi.v2 train --config ...` and `python -m tdi.v2 evaluate --model-dir ...`

---

## Documentation Index

- **[Pipeline Technical Reference](file:///Users/skn506/Documents/Claude/Projects/3di-new/docs/pipeline.md)**: Detailed reference guide covering data processing specifications, model architecture parameters, export file formats, and evaluation metrics.
- **[Experiments Quarantine](file:///Users/skn506/Documents/Claude/Projects/3di-new/experiments/README.md)**: Quarantined, runnable snapshots of self-contained experimental components (rotation trick gradient, coordinate augmentation, alignment batch sampler).
- **[Historical Design Archive](file:///Users/skn506/Documents/Claude/Projects/3di-new/docs/archive/)**: Historical design proposals, tier plans, and verification reports from earlier phases of the v2 overhaul.
