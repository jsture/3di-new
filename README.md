# 3Di VAE v2: a single, auditable structural-alphabet learner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

One reliable path from aligned residue descriptors to a trained discrete 3Di-style structural
alphabet: a config-driven preprocessing stage, a plain-PyTorch training stage, and an evaluation
stage that emits a substitution matrix and alphabet diagnostics.

---

## The v2 pipeline

Three stages, each a separate command:

1. **Build features** (`python -m tdi.data build-features`) — parse PDBs, expand CIGAR
   alignments, filter residue pairs by superposed Cα distance, and write standardized feature
   arrays + a train-only scaler + auditable per-pair metadata.
2. **Train** (`python -m tdi.v2 train`) — train exactly one quantizer into a self-describing
   run directory.
3. **Evaluate** (`python -m tdi.v2 evaluate`) — encode validation structures to sequences and
   compute the substitution matrix + alphabet diagnostics.

### The model

An MLP encoder maps each 10-D residue descriptor to a latent; a quantizer forms the discrete
state; an MLP decoder predicts the **aligned partner's** descriptors. Training minimizes a single
`smooth_l1` partner-prediction loss plus the quantizer loss, with a **straight-through** gradient,
**fp32** throughout, plain PyTorch (no Lightning), and a **fixed learning rate** by default.

Two quantizers sit behind one interface, selected by `--quantizer {vq,fsq}` — a run trains exactly
one:

- **EMA-VQ** (reference): EMA codebook updates, commitment loss, L2-normalized (cosine) lookup,
  mandatory dead-code replacement, one-shot k-means init.
- **FSQ `[5,4]`** (comparator): a fixed finite-scalar grid, no learned codebook, no collapse to
  guard against.

`n_states` is configurable and **capped at 50** (the alphabet has 50 letters; more needs a longer
alphabet). Default is 20 — for FSQ that is levels `[5, 4]`. The alphabet is recorded in the export
`config.json`, so encode/eval never hardcode it.

The optional, standalone `scripts/compare_quantizers.py` driver runs the normal train + evaluate
path twice (once `vq`, once `fsq`) and writes a side-by-side `comparison_report.json`. It is not
part of the core path — the core never imports it; reach for it only when you want the comparison.

---

## Getting started

```bash
uv sync
```

### 1. Build features

```bash
uv run python -m tdi.data build-features --config configs/data/scop.yaml --force
```

Writes a processed dataset (`train_x_raw.npy`/`train_y_raw.npy`, `val_*`, `scaler.npz`, lean
`*_metadata.parquet`, `structures.parquet`, `report.json` + `report.md`, `manifest.json`,
`DATACARD.md`). Add `--full-report` to also emit the sequence-separation and Cα-distance histograms.

### 2. Train one quantizer

```bash
# EMA-VQ (reference)
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml --quantizer vq --out runs/ema_vq
# FSQ [5,4] (comparator)
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml --quantizer fsq --out runs/fsq_5x4
```

Override any config field with dotted flags, e.g. `--model.n_states 24`, `--model.levels "[5,5]"`,
`--train.max_epochs 10`, `--train.scheduler cosine`. The run directory gets `encoder_state_dict.pt`,
`decoder_state_dict.pt`, `config.json`, `scaler.json`, `centroids.npy` (vq) or `fsq_levels.json`
(fsq), plus `run_config.resolved.json` and `train_log.csv`.

### 3. Evaluate

```bash
uv run python -m tdi.v2 evaluate \
  --model_dir runs/ema_vq \
  --pdb_dir data/pdb \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out_dir runs/ema_vq/eval
```

Outputs:
- `sequences.txt` — one encoded 3Di sequence per structure (invalid residues render as the
  configured `invalid_state`, default `X`).
- `submat.txt` — the log-odds substitution matrix over the alphabet.
- `evaluation_report.json` — mutual information (`mi`, `mi_tot`), state usage, `dead_state_fraction`,
  and `normalized_entropy`.

---

## Testing

```bash
uv run pytest
```

---

## Directory structure

```
├── configs/            # YAML configs for data generation and model training
├── data/
│   ├── raw/            # Baseline SCOPe SIDs and alignment pairfiles
│   ├── derived/        # Train/val split files
│   └── processed/      # Preprocessed float32 feature arrays
├── scripts/            # Splits, structure fetch, and the optional compare_quantizers driver
├── src/tdi/
│   ├── data/           # Preprocessing pipeline orchestration
│   └── v2/             # Encoder/decoder, quantizers, training & evaluation
└── tests/              # Unit and integration tests
```

Removed objectives from earlier iterations — GaussianNLL, contrastive learning, self-reconstruction,
the warmup curriculum, the transition head, the rotation-trick gradient, and coordinate/descriptor
augmentation — live in git history (see `experiments/README.md`).
