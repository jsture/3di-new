# 3Di VAE v2 — a discrete structural alphabet learner

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

Learns a **3Di-style structural alphabet**: a small set of discrete states (default 20) that each
residue of a protein structure is mapped to, so that structures can be compared as *sequences*
instead of as coordinates.

The training signal comes from structural alignments. Two residues that a structural aligner put
in the same alignment column should look alike, so the model encodes residue *i* of one structure
into a discrete state and is asked to reconstruct the local geometry of its aligned partner *j* in
the other structure. States that survive this are states that generalize across homologs.

Everything runs as three explicit commands — build features, train, evaluate — with YAML configs,
content-hashed dataset manifests, and self-describing run directories.

---

## Quickstart

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

### 0. Get structures (once)

Downloads the Foldseek SCOPe40 benchmark archive and lays it out as one file per SCOP domain ID
under `data/external/foldseek_scop40/pdb_by_sid/` (~11k structures).

```bash
uv run python scripts/fetch_scop40_structures.py --out-dir data/external/foldseek_scop40
```

The alignment pairfiles (`data/derived/pairfiles/tmaln-06.{train,val}.out`) and the SCOP lookup
(`data/raw/scop_lookup.tsv`) are already in the repo.

### 1. Build features

```bash
uv run python -m tdi.data build-features --config configs/data/scop.yaml --force
```

Parses PDBs, expands the alignment CIGARs, drops residue pairs whose superposed Cα distance
exceeds `max_ca_dist`, fits a **train-only** standardizer, and writes a processed dataset to
`data/processed/scop_ca5_v1/`. Add `--full-report` for the sequence-separation and Cα-distance
histograms.

### 2. Train one quantizer

A run trains exactly one quantizer — pick it with `--quantizer`.

```bash
# EMA-VQ, the reference learner
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml --quantizer vq --out runs/ema_vq

# FSQ [5,4], the fixed-grid comparator
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml --quantizer fsq --out runs/fsq_5x4

# Random-init VQ, the "no k-means" floor
uv run python -m tdi.v2 train --config configs/train/scop_vq_baseline.yaml --out runs/vq_baseline_random
```

#### Schedule-Free AdamW (optional)

`--optimizer schedulefree` swaps AdamW for Schedule-Free AdamW ([Defazio et al., NeurIPS
2024](https://arxiv.org/abs/2405.15682)), which drops the need to pick a stopping horizon:

```bash
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml \
  --optimizer schedulefree --train.lr 0.0025 --out runs/sf_vq
```

It replaces the LR schedule rather than composing with one, so pairing it with
`train.scheduler: cosine` is rejected at config load. The paper reports its best learning rates
are **larger** than the base optimizer's, so sweep `--train.lr` rather than reusing the AdamW
value. Warmup is still required and stays exposed as `train.sf_warmup_steps`.

Internally the optimizer keeps two views of the weights — the point gradients are taken at, and
the averaged point it is defined to return. The trainer switches to the averaged view before
validation, before every best-checkpoint snapshot, and before the export, so run directories
always hold the returned iterate.

### 3. Evaluate

Encodes the validation structures to sequences, then scores the alphabet against the alignments.

```bash
uv run python -m tdi.v2 evaluate \
  --model-dir runs/ema_vq \
  --pdb-dir data/external/foldseek_scop40/pdb_by_sid \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out-dir runs/ema_vq/eval
```

The virtual-center geometry is read from the run's `config.json`. If that run was trained from a
dataset whose manifest carried no provenance, `virtual_center` is `null` there and you must pass it
explicitly: `--virt 270 0 2` (matching `features.virtual_center` in the data config).

---

## How it works

### Descriptors (10-D per residue)

| Columns | Meaning |
| --- | --- |
| 0–6 | Cosines of 7 angles between backbone tangents and chord vectors to the nearest spatial neighbour |
| 7 | Cα–Cα distance (Å) to that nearest neighbour |
| 8 | Sequence separation, clamped to [−4, +4] |
| 9 | Signed log separation, `sign(Δ)·ln(|Δ|+1)`, with `Δ = j − i` |

"Nearest neighbour" is found from a **virtual center** placed relative to the backbone by
`features.virtual_center: [alpha, beta, d]` — the same convention Foldseek's 3Di uses.

### Model

`descriptor(i) → encoder → z → quantizer → z_q → decoder → descriptor(j)`

- Encoder: residual MLP, `input_dim=10 → hidden_dim=64 → z_dim=4`, depth 3.
- Decoder: residual MLP, `z_dim → hidden_dim → input_dim`, depth 2. It predicts the **aligned
  partner's** descriptors, never its own input.
- Loss: one `smooth_l1` partner-prediction term plus the quantizer loss. EMA-VQ uses a
  straight-through gradient.
  Training is fp32 throughout, plain PyTorch (no Lightning), with a fixed LR by default.
- Early stopping on `val_loss` with `patience`; the best weights are restored before export.

### The two quantizers

| | `--quantizer vq` (EMA-VQ) | `--quantizer fsq` (FSQ) |
| --- | --- | --- |
| Codebook | Learned, EMA-updated | None — a fixed scalar grid |
| Lookup | Cosine (L2-normalized) | Rounding per dimension |
| Gradient | Straight-through | Bounding derivative + rounding STE |
| Collapse risk | Real; guarded by mandatory dead-code replacement (after `replacement_warmup_steps`) | None by construction |
| Init | One-shot k-means (`kmeans_init`) | n/a |
| States | `n_states` | `prod(levels)`, and `z_dim = len(levels)` |

FSQ **overrides** `n_states` and `z_dim` from `levels` (default `[5, 4]` → 20 states, `z_dim=2`).

`n_states` is capped at **50**, the length of the alphabet
`ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz`; more states needs a longer alphabet. The
alphabet and the invalid-residue character (`X`) are recorded in the export `config.json`, so
encoding and evaluation never hardcode them.

---

## What each stage writes

**`build-features` → `data/processed/<name>/`**

| File | Contents |
| --- | --- |
| `{train,val}_{x,y}_raw.npy` | Paired input/target descriptors (bidirectional: each aligned pair contributes both directions) |
| `scaler.npz` | Feature mean/std, fit on train pairs only — no validation leakage |
| `{train,val}_metadata.parquet` | Per-pair audit trail: source alignment row, SCOP classification |
| `structures.parquet` | Per-structure QC: residue counts, valid fraction, missing-backbone flags |
| `report.json` / `report.md` | Filter attrition, stage counts, optional histograms |
| `manifest.json` | Resolved config + SHA-256 of every input and output |
| `DATACARD.md` | Human-readable provenance card |

**`train` → the run directory**

| File | Contents |
| --- | --- |
| `encoder_state_dict.pt`, `decoder_state_dict.pt` | Weights |
| `config.json` | Self-describing export: dims, quantizer, alphabet, `invalid_state`, geometric provenance |
| `scaler.json` | Standardization params, so inference is standalone |
| `centroids.npy` (vq) *or* `fsq_levels.json` (fsq) | The discrete codebook |
| `run_config.resolved.json` | Fully resolved training config |
| `train_log.csv` | Total, reconstruction, and quantizer losses plus state diagnostics per epoch |

**`evaluate` → `--out-dir`**

| File | Contents |
| --- | --- |
| `sequences.txt` | `<sid> <sequence>` per structure; invalid residues become `invalid_state` (`X`) |
| `submat.txt` | Log-odds substitution matrix over the alphabet |
| `evaluation_report.json` | `mi`, `mi_tot`, `state_usage`, `dead_state_fraction`, `normalized_entropy`, and encoding failure counts |

Read `mi` / `mi_tot` as *how much a state in one structure tells you about the aligned state in
the other* — higher is a more discriminative alphabet. `dead_state_fraction` near 0 and
`normalized_entropy` near 1 mean the alphabet is actually using all its letters.

---

## Configuration

Configs live in `configs/`. Every field can be overridden on the command line with a dotted flag,
and the resolved result is what gets recorded:

```bash
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml \
  --model.n_states 24 --train.max_epochs 10 --train.scheduler cosine
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml \
  --quantizer fsq --model.levels "[5,5]"
```

Knobs worth knowing:

| Key | Default | Notes |
| --- | --- | --- |
| `model.quantizer` | `vq` | `vq` or `fsq` |
| `model.n_states` | `20` | ≤ 50; ignored by FSQ (derived from `levels`) |
| `model.levels` | `null` | FSQ grid; defaults to `[5, 4]` |
| `model.z_dim` | `4` | Latent width; FSQ pins it to `len(levels)` |
| `model.loss` | `smooth_l1` | or `mse` |
| `model.commitment_cost` | `0.25` | VQ commitment penalty |
| `model.decay` | `0.99` | VQ EMA decay |
| `model.min_count` | `1.0` | VQ dead-code threshold |
| `train.scheduler` | `none` | `none` (fixed LR) or `cosine`; must stay `none` under Schedule-Free |
| `train.optimizer` | `adamw` | or `schedulefree` (Schedule-Free AdamW) |
| `train.sf_warmup_steps` | `500` | Schedule-Free linear LR warmup; ignored by `adamw` |
| `train.sf_beta` | `0.9` | Schedule-Free momentum interpolation; ignored by `adamw` |
| `train.patience` | `5` | Early stopping on `val_loss` |
| `train.kmeans_init` | `true` | VQ codebook seeding; `false` gives the random-init baseline |
| `features.max_ca_dist` | `5.0` | Å; pair filter in the data config |
| `sampling.max_pairs_per_alignment` | `768` | Stops long alignments dominating the set |

Unknown sections or keys are rejected at load time rather than silently ignored.

The data CLI has two more subcommands: `python -m tdi.data validate` (structure QC and
CIGAR-semantics checks) and `python -m tdi.data report` (re-render `report.md` from `report.json`).

---

## Comparing VQ against FSQ

`scripts/compare_quantizers.py` runs the normal train + evaluate path twice and tabulates the two
`evaluation_report.json` side by side:

```bash
uv run python scripts/compare_quantizers.py \
  --config configs/train/scop_v2_default.yaml \
  --pdb-dir data/external/foldseek_scop40/pdb_by_sid \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out-root runs/compare --virt 270 0 2
```

Writes `comparison_report.json` and `comparison.md` under `--out-root`. This is deliberately
opt-in and standalone — the core `tdi.v2` code never imports it. Exactly two runs, no sweeps.

---

## Development

```bash
uv run pytest
uv run ruff check src/tdi
uv run ruff format --check src/tdi
uv run pyright src/tdi
```

`pre-commit` hooks are configured in `.pre-commit-config.yaml`; CI (`.github/workflows/ci.yml`)
runs lint, type-check, and the test suite on every push to `main` and every PR.

---

## Repository layout

```
configs/
  data/       # preprocessing config (SCOPe baseline)
  train/      # training configs (default + random-init VQ baseline)
data/
  raw/        # SCOPe SIDs, lookup table, source alignments
  derived/    # train/val pairfile splits
  external/   # fetched SCOPe40 structures (pdb_by_sid/)
  processed/  # built feature arrays + scaler + manifests
docs/         # architecture overview and the pipeline technical reference
experiments/  # quarantined runnable snapshots of removed mechanisms
scripts/      # structure fetch, splits, optional quantizer comparison
src/tdi/
  data/       # preprocessing: PDB parsing, CIGAR expansion, QC, datacards
  v2/         # model, quantizers, training loop, encoding, evaluation
tests/        # unit, integration, and golden tests
runs/         # training run directories (gitignored outputs)
```

More detail: [`docs/pipeline.md`](docs/pipeline.md) is the stage-by-stage technical reference.

## History

Objectives removed during the v2 simplification — GaussianNLL, contrastive learning,
self-reconstruction, the warmup curriculum, the transition head, and coordinate/descriptor
augmentation — are recoverable from git history. The self-contained ones are kept runnable under
`experiments/`; see [`experiments/README.md`](experiments/README.md) for how to retrieve the rest.

## License

GPL-3.0-or-later — see [`LICENSE`](LICENSE). Copyright (C) 2026 Jakob.

This repository descends from [foldseek-analysis](https://github.com/steineggerlab/foldseek-analysis)
(Steinegger lab); its early history contains that project's code, and the original 3Di training
scripts under `training/` were GPL-licensed. `src/tdi/` is a rewrite, but the project stays under
GPLv3 to remain compatible with that lineage.
