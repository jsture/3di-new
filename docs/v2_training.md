# Training the v2 3Di Model

This document describes the end-to-end workflow for preparing data, validating the pipeline, training the modernized VQ-VAE (v2) model, and evaluating the resulting discrete structural alphabet.

## 1. Expected Data Layout

The repository reads from the following data layout structure under the root directory:

```text
data/
  raw/
    pdbs_train.txt
    pdbs_val.txt
    scop_lookup.tsv
    tmaln-06.out

  derived/
    pairfiles/
      tmaln-06.train.out
      tmaln-06.val.out

  external/
    foldseek_scop40/
      pdb_by_sid/
        d1xxxx_
        d2yyyy_
        ...
```

- **`data/raw/`**: Fixed SCOPe baseline alignments and lookups.
- **`data/derived/pairfiles/`**: Contains train/validation alignment splits created by separating raw alignments based on corresponding training and validation domains. Do not train directly from `data/raw/tmaln-06.out`.

---

## 2. Generate Derived Pairfiles

To recreate or extract the derived train and validation pairfiles, run the following awk split commands from the repository root:

```bash
mkdir -p data/derived/pairfiles

# Extract unique Domain IDs (SIDs) from split configuration lists
LC_ALL=C awk 'NF {print $1}' data/raw/pdbs_train.txt | sort -u > /tmp/train_sids.txt
LC_ALL=C awk 'NF {print $1}' data/raw/pdbs_val.txt   | sort -u > /tmp/val_sids.txt

# Extract train-specific alignments where both domains belong to train set
awk 'NR==FNR {train[$1]=1; next} ($1 in train) && ($2 in train)' \
  /tmp/train_sids.txt data/raw/tmaln-06.out \
  > data/derived/pairfiles/tmaln-06.train.out

# Extract validation-specific alignments where both domains belong to validation set
awk 'NR==FNR {val[$1]=1; next} ($1 in val) && ($2 in val)' \
  /tmp/val_sids.txt data/raw/tmaln-06.out \
  > data/derived/pairfiles/tmaln-06.val.out
```

Expected alignment counts for the default splits:
- `data/raw/tmaln-06.out`: ~24,525 lines
- `data/derived/pairfiles/tmaln-06.train.out`: ~20,657 lines
- `data/derived/pairfiles/tmaln-06.val.out`: ~3,814 lines

---

## 3. Build Training Arrays (Data Preprocessing)

Features are extracted from aligning residue-descriptor pairs using the configuration file [configs/data/scop.yaml](file:///Users/skn506/Documents/Claude/Projects/3di-new/configs/data/scop.yaml). This step:
1. Expands CIGAR strings to establish residue-to-residue correspondences.
2. Filters out invalid coordinates/backbones.
3. Filters by $C_\alpha$ Euclidean distances (maximum distance threshold, e.g., $5.0\text{ \AA}$).
4. Standardizes input features (fitting scale parameters *only* on the training split).
5. Reports skipped alignments to TSV outputs to keep track of any malformed structures.

Run the preprocessing command:
```bash
uv run python -m tdi.data build-features --config configs/data/scop.yaml
```

### CLI Overrides
You can override YAML configuration settings on the command line, for example:
```bash
uv run python -m tdi.data build-features \
  --config configs/data/scop.yaml \
  --out_dir data/processed/scop_custom \
  --max_ca_dist 6.0 \
  --max_pairs 512
```

### Output Layout
The preprocessing command populates the output directory (default: `data/processed/scop_ca5_v1/`) with the following files:
```text
data/processed/scop_ca5_v1/
  train_x_raw.npy            # Unstandardized training input features
  train_y_raw.npy            # Unstandardized training partner features
  val_x_raw.npy              # Unstandardized validation input features
  val_y_raw.npy              # Unstandardized validation partner features
  scaler.npz                 # Fitted standardizer parameters (mean and std)
  train_metadata.parquet     # Row-aligned metadata (e.g., domain/residue IDs) for train
  val_metadata.parquet       # Row-aligned metadata for validation
  structures.parquet         # Summary QC statistics for each PDB structure
  train_skipped_alignments.tsv # TSV log listing any skipped training alignments and error reasons
  val_skipped_alignments.tsv   # TSV log listing any skipped validation alignments and error reasons
  report.json                # Summary metrics and statistics JSON
  report.md                  # Human-readable markdown validation summary report
  DATACARD.md                # Dataset description manifest
  manifest.json              # Version tracking and files integrity checksums
```

---

## 4. Preprocessing Quality Control & Validation

The `tdi.data` CLI provides commands for checking data health and re-rendering reports:

### Validate Dataset Integrity
Verifies the semantic structure and alignment consistency of the produced processed directory:
```bash
uv run python -m tdi.data validate --config configs/data/scop.yaml
```

### Re-render Preprocessing Reports
Rebuilds the markdown report from the `report.json` data:
```bash
uv run python -m tdi.data report --config configs/data/scop.yaml
```

---

## 5. Model Configuration

Model configurations are defined in [configs/train/scop_v2_default.yaml](file:///Users/skn506/Documents/Claude/Projects/3di-new/configs/train/scop_v2_default.yaml). The configuration schema maps to dataclasses in [src/tdi/v2/train_config.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/train_config.py).

Modern default settings in the schema prioritize high codebook usage and robust gradient flow:
- **`model.quantizer_type`**: `"ema_vq"` (Exponential Moving Average Vector Quantizer).
- **`quantizer.gradient_mode`**: `"rotation_trick"` (Householder reflection surrogate gradient).
- **`quantizer.kmeans_init`**: `True` (Seeding centroids using warmed-up latents via K-Means).
- **`training.quantizer_warmup_epochs`**: `1` (Continuous latent warmup before quantization is enabled).
- **`loss.lambda_self` / `lambda_contrast` / `lambda_usage`**: Set to `0.05`, `0.02`, and `0.001` respectively, ensuring aligned feature clustering without code collapse.

---

## 6. Train the Model

To train the VQ-VAE v2 model using the default configuration file:
```bash
uv run python -m tdi.v2 train --config configs/train/scop_v2_default.yaml
```

### Dotted Overrides
Any nested parameter in the `TrainConfig` schema can be overridden at runtime using dotted notation flags:
```bash
uv run python -m tdi.v2 train \
  --config configs/train/scop_v2_default.yaml \
  --training.max_epochs 10 \
  --quantizer.gradient_mode ste \
  --optimizer.lr 0.002
```

### Training Pipeline Phases
1. **Warmup Phase**: The model is fit with continuous latents for the configured `quantizer_warmup_epochs` (default: 1 epoch). The quantization discretization step is bypassed.
2. **K-Means Initialization**: At the boundary of the warmup epoch, the codebook is initialized by collecting encoder outputs across dataloader batches and clustering them into starting centroids.
3. **Surrogate Gradients**: Training proceeds with discrete states. Forward lookups map outputs to discrete centroids, and gradients are propagated to the encoder via either standard straight-through estimation (`ste`) or the angular-preserving `rotation_trick` (recommended default).
4. **Early Stopping**: The run automatically monitors validation performance and exits early if validation scores plateau.
5. **Export**: The best model checkpoint is loaded, features are frozen, and final artifacts are exported.

---

## 7. Model Export Layout

A completed training run creates the output directory (default: `outputs/models/scop_v2_default_seed1/`) containing:
```text
outputs/models/scop_v2_default_seed1/
  best-checkpoint-epoch=XX-val_score=YY.ckpt   # PyTorch Lightning raw checkpoint
  training_config.yaml                          # Record of the resolved runtime configuration
  encoder_state_dict.pt                         # Frozen encoder weight parameters
  model_config.json                             # Target dimensions and hyperparameter values
  feature_scaler.json                           # Preprocessing mean and std scaling parameters
  centroids.npy                                 # VQ codebook embeddings matrix
```

Downstream encoding, feature representation, and evaluation utilities consume the sub-artifacts exported to the output folder.

---

## 8. Downstream Alphabet Evaluation

To evaluate the quality of the discrete structural alphabet, use the `evaluate` subcommand:
```bash
uv run python -m tdi.v2 evaluate \
  --model_dir outputs/models/scop_v2_default_seed1 \
  --pdb_dir data/external/foldseek_scop40/pdb_by_sid \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out_dir outputs/evaluations/scop_v2_default_seed1 \
  --virt 270.0 0.0 2.0
```

### Evaluation Outputs
This script translates aligned structures into discrete state sequences and outputs:
- **`sequences.txt`**: Encoded structural alphabet sequence for each Domain ID.
- **`submat.txt`**: Computed log-odds substitution scoring matrix (similar to BLOSUM/BLAST).
- **`evaluation_report.json`**: Detailed metrics reporting mutual information (MI) and transition-adjusted mutual information ($MI_{tot}$). Higher mutual information suggests better capture of alignment syntax.

---

## 9. Reproducibility & Multiple Seed Training

To run multiple randomized seeds for replication or robust model ensembling, execute a script loop:
```bash
for seed in 1 2 3 4 5; do
  uv run python -m tdi.v2 train \
    --config configs/train/scop_v2_default.yaml \
    --training.seed ${seed} \
    --outputs.out_dir outputs/models/scop_v2_default_seed${seed}
done
```

---

## 10. Diagnostics and Sanity Checks

### Check Preprocessed Arrays
Ensure the generated preprocessed dataset is finite and consistent:
```python
import numpy as np
from pathlib import Path

data_dir = Path("data/processed/scop_ca5_v1")
for name in ["train_x_raw.npy", "train_y_raw.npy", "val_x_raw.npy", "val_y_raw.npy"]:
    arr = np.load(data_dir / name)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, finite={np.isfinite(arr).all()}")
```

### Check Exported Scaling Parameters
Confirm standardizer parameters match expected ranges:
```python
import json
from pathlib import Path

scaler_path = Path("outputs/models/scop_v2_default_seed1/feature_scaler.json")
with open(scaler_path) as f:
    scaler = json.load(f)
print("Scaler Mean:", scaler["mean"])
print("Scaler Std:", scaler["std"])
```
