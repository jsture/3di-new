# 3Di VAE v2 Pipeline Technical Reference

This document provides a reference for the 3-stage structural alphabet pipeline.

---

## 1. Feature Preprocessing (`tdi.data`)

### Overview
Extracts 10-dimensional local structural descriptors from PDB coordinates and expands alignment pairfiles (using the custom `P`=aligned-pair CIGAR convention).

```bash
uv run python -m tdi.data build-features --config configs/data/scop.yaml [--force]
```

### Feature Definition
Each residue is described by a 10-D descriptor vector:
- **Columns 0–6**: Cosines of 7 inter-residue angles between backbone tangents and chord vectors to the nearest structural neighbor.
- **Column 7**: Distance $d$ in Angstroms between $C_\alpha$ positions of the residue and its nearest spatial neighbor.
- **Column 8**: Sequence separation clamped to $[-4, +4]$.
- **Column 9**: Signed logarithmic sequence delta: $\text{sign}(\Delta) \cdot \ln(|\Delta| + 1)$ where $\Delta = j - i$ (partner index minus source index).

### Data Artifacts Emitted
- `train_x_raw.npy` / `train_y_raw.npy`: Aligned bidirectional input/target feature pairs for training.
- `val_x_raw.npy` / `val_y_raw.npy`: Aligned bidirectional pairs for validation.
- `scaler.npz`: Feature mean and standard deviation fit strictly on training pairs (no validation leakage).
- `train_metadata.parquet` / `val_metadata.parquet`: Per-pair audit metadata including source alignment rows and SCOP classifications.
- `structures.parquet`: Structure-level quality control table (residue counts, valid fraction, missing backbone flags).
- `report.json` / `report.md`: Preprocessing audit report with filter attrition metrics.
- `DATACARD.md`: Self-contained provenance card with SHA-256 digests.
- `manifest.json`: Cryptographic manifest recording input/output checksums and runtime configuration.

---

## 2. Model Training (`tdi.v2.train`)

### Overview
Trains a single discrete structural alphabet quantizer using plain PyTorch.

```bash
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml --quantizer vq --out runs/ema_vq
```

### Quantizer Backends
1. **EMA-VQ (`--quantizer vq`)**:
   - Continuous latents mapped to $K$ codebook centroids via cosine distance.
   - Codebook updated using Exponential Moving Average (EMA).
   - Mandatory dead-code replacement replacing inactive centroids with active batch latents.
   - Initialized with one-shot k-means clustering on initial encoder representations.
2. **FSQ (`--quantizer fsq`)**:
   - Fixed scalar grid with defined dimension levels (e.g. $[5, 4]$ for 20 states).
   - No codebook collapse or learned cluster updates.

### Output Artifacts
- `encoder_state_dict.pt`: Trained MLP encoder parameters.
- `decoder_state_dict.pt`: Trained MLP partner-prediction decoder parameters.
- `config.json`: Self-describing export config recording alphabet definition, quantizer type, and geometric provenance.
- `scaler.json`: Exported standardization parameters.
- `centroids.npy` (VQ) or `fsq_levels.json` (FSQ): Discrete codebook definitions.
- `train_log.csv`: Training and validation loss curves, perplexity, and dead state counts.

---

## 3. Alphabet Evaluation (`tdi.v2.cli`)

### Overview
Encodes test PDB structures into 3Di sequences and calculates state transition substitution matrices and mutual information diagnostics.

```bash
uv run python -m tdi.v2 evaluate \
  --model-dir runs/ema_vq \
  --pdb-dir data/pdb \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out-dir runs/ema_vq/eval \
  --virt 270 0 2
```

### Emitted Diagnostics
- `sequences.txt`: Encoded 3Di character sequences for all requested structures.
- `submat.txt`: Log-odds substitution scoring matrix over the alphabet states.
- `evaluation_report.json`: Metrics including raw mutual information (`mi`), transition-adjusted mutual information (`mi_tot`), state usage frequencies, normalized entropy, and dead state fraction.
