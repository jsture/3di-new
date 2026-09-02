# Dataset-independent training-data recommendations for robust discrete structural states

These recommendations apply regardless of whether the source corpus is SCOPe, CATH, ECOD, PDB, AlphaFoldDB, ESM Atlas, or another structural collection. They assume only that the final training examples are aligned structural descriptor pairs of the form:

```text
x_i = descriptor for residue i in structure/domain A
y_j = descriptor for aligned residue j in structure/domain B
```

The objective is to produce a discrete alphabet whose states are robust, well-used, and useful for downstream substitution statistics or alignment.

---

## 1. Split the corpus before generating residue-level examples

### Actionable steps

1. Assign every structure/domain to a leakage-control group before pair extraction.
2. Use the strongest available grouping key:
   - fold/topology ID, if available;
   - superfamily/homology group, if available;
   - sequence cluster ID;
   - structure cluster ID;
   - otherwise, source structure/domain ID.
3. Create train/validation/test partitions at the group level.
4. Generate residue-pair examples separately for each partition.
5. Reject alignments that cross partitions.

### Reasoning

Residue pairs are not independent samples. Randomly splitting residue pairs allows almost identical local environments from the same structure pair to appear in both training and validation. That makes validation loss and state usage look better than true generalization.

### Implementation note

Use a manifest file before preprocessing:

```text
structure_id,domain_id,group_id,split
```

Then require `create_training_data.py` to accept `--split train`, `--split val`, or an explicit `--sid-list` / `--domain-list`.

---

## 2. Expand alignments into an auditable residue-pair table

### Actionable steps

1. Keep compact alignment strings such as CIGAR for reproducibility.
2. Also materialize an expanded table with one row per aligned residue pair.
3. Store at least:

```text
source_corpus
structure_id_1
structure_id_2
domain_id_1
domain_id_2
residue_index_1
residue_index_2
alignment_id
alignment_score
global_quality_score, if available
local_quality_score, if available
ca_pair_distance, if computable
valid_descriptor_1
valid_descriptor_2
split
group_id_1
group_id_2
```

4. Save the descriptor arrays separately, keyed by row index, or save them in a compressed array file with the same row ordering.

### Reasoning

A single `.npy` array containing only `x` and `y` makes failures hard to diagnose. Provenance is needed to identify bad structures, bad alignments, overrepresented groups, and rare-state sources.

### Recommended file formats

Use one of:

```text
training_pairs.parquet     # best for metadata tables
training_pairs.npz         # simple NumPy arrays
training_pairs.zarr        # chunked scalable arrays
training_pairs.arrow       # columnar metadata and arrays
```

---

## 3. Filter aligned residue pairs by local structural consistency

### Actionable steps

1. Compute the aligned Cα–Cα distance for every residue pair after structural superposition, when coordinates and transformation are available.
2. Exclude residue pairs above a strict distance threshold.
3. Use this default threshold:

```text
ca_pair_distance <= 5.0 Å
```

4. Store the raw distance even for excluded pairs in the preprocessing report.
5. If the alignment source does not provide a superposition, store a missing value and mark the row as `distance_unavailable`; do not silently treat it as valid.

### Reasoning

The model is trained to predict the aligned partner descriptor. A poor residue-level alignment gives noisy supervision: it tells the model that two geometries should predict each other even when they are not local structural counterparts. This directly degrades the learned discrete states.

### Example function

```python
import numpy as np


def ca_distance_mask(ca1: np.ndarray, ca2: np.ndarray, max_dist: float = 5.0) -> np.ndarray:
    """Return mask for aligned residue pairs within max_dist Angstrom."""
    dist = np.linalg.norm(ca1 - ca2, axis=1)
    return dist <= max_dist
```

---

## 4. Enforce descriptor validity after all filters

### Actionable steps

1. Compute descriptor-validity masks for both structures.
2. Require valid local context for both aligned residues.
3. Apply filters in this order:

```text
alignment membership
residue index in bounds
valid local coordinates
valid descriptor for residue 1
valid descriptor for residue 2
local structural consistency filter
quality/confidence filter
```

4. Fail the preprocessing run if any `NaN` or `Inf` remains after filtering.

### Reasoning

Descriptor validity can change when feature definitions change. Filtering should happen after descriptor extraction, not only during upstream alignment generation.

### Example check

```python
import numpy as np


def assert_finite_features(x: np.ndarray, name: str) -> None:
    if not np.isfinite(x).all():
        bad = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{name} contains {bad} non-finite values")
```

---

## 5. Standardize features using training statistics only

### Actionable steps

1. Fit feature mean and standard deviation on the training split only.
2. Apply the same transform to training, validation, test, and inference inputs.
3. Export the scaler with the model artifacts.
4. Reject features with near-zero variance unless they are intentionally constant.

### Reasoning

Structural descriptors often mix cosines, distances, clipped sequence distances, and log sequence distances. These dimensions have different scales. Without standardization, large-scale dimensions can dominate the loss and the learned state boundaries.

### Concrete implementation

```python
import numpy as np


def fit_standardizer(x_train: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)
```

`sklearn.preprocessing.StandardScaler` implements the same standardization idea by removing the training-set mean and scaling to unit variance.

---

## 6. Save preprocessing configuration as a versioned artifact

### Actionable steps

Save a `preprocessing_config.json` with:

```text
feature_version
alignment_parser_version
coordinate_parser_version
filters and thresholds
standardization mean/std file path
split manifest path
random seed
source manifest checksum
creation timestamp
software versions
```

### Reasoning

The learned states are only interpretable relative to the exact preprocessing pipeline. If a feature sign convention, coordinate filter, or scaling rule changes, the centroids and downstream substitution matrix are no longer directly comparable.

---

## 7. Balance training examples by source group and alignment

### Actionable steps

1. Compute sample weights before training.
2. Downweight overrepresented groups such as large folds, long structures, dense alignments, or duplicated clusters.
3. Use one or more of these weights:

```text
1 / number_of_examples_from_alignment
1 / number_of_examples_from_structure_pair
1 / number_of_examples_from_group
1 / cluster_size
```

4. Use weighted sampling during training rather than permanently deleting all redundant examples.

### Reasoning

Uniform residue-pair sampling gives more influence to long structures and overrepresented families. The resulting alphabet can encode corpus frequency rather than broadly useful structural states.

### PyTorch function

Use `torch.utils.data.WeightedRandomSampler` through `torch.utils.data.DataLoader`.

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

loader = DataLoader(
    dataset,
    batch_size=512,
    sampler=sampler,
    drop_last=True,
)
```

---

## 8. Cap residue pairs per alignment during dataset construction

### Actionable steps

1. Set a maximum number of residue-pair examples per structure/domain pair.
2. Use a deterministic random seed for subsampling.
3. Store both pre-cap and post-cap counts.

Recommended default:

```text
max_pairs_per_alignment = 512 or 1024
```

### Reasoning

Very long alignments can dominate training. Capping prevents a small number of large pairs from overwhelming the state-learning objective.

### Example

```python
import numpy as np


def cap_indices(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))
```

---

## 9. Balance or at least report contact-type composition

### Actionable steps

For every source residue, compute and store:

```text
nearest_neighbor_sequence_separation = abs(i - nearest_neighbor_i)
nearest_neighbor_distance
contact bin
```

Use bins such as:

```text
local:       |i - j| <= 1
near-local:  2 <= |i - j| <= 4
medium:      5 <= |i - j| <= 16
long-range: |i - j| > 16
```

Either:

1. sample approximately evenly across bins, or
2. report the bin distribution and decide whether reweighting is required.

### Reasoning

A structural alphabet intended to encode tertiary interactions should not be dominated by local sequence neighbors. Contact composition should be measured and controlled independently of the source dataset.

---

## 10. Add structure-level quality filters

### Actionable steps

Apply filters before alignment expansion and descriptor extraction:

```text
minimum length
maximum missing-coordinate fraction
minimum valid-descriptor fraction
maximum fraction of non-standard residues, if relevant
minimum experimental/model confidence, if available
```

Suggested defaults:

```text
length >= 40 residues
valid_descriptor_fraction >= 0.80
missing_CA_fraction <= 0.05
```

For predicted structures with confidence scores:

```text
mean local confidence above project threshold
exclude residues below local-confidence threshold
exclude long low-confidence intervals
```

### Reasoning

Bad structures generate invalid or noisy descriptors. The alphabet should encode stable structural geometry, not missing-coordinate artifacts or low-confidence disorder.

---

## 11. Add training-only geometric augmentation

### Actionable steps

1. Add optional coordinate noise before descriptor extraction for training only.
2. Do not apply this to validation/test data.
3. Use small noise magnitudes:

```text
coordinate noise standard deviation = 0.05–0.20 Å
```

4. Record augmentation parameters in the preprocessing config.

### Reasoning

A robust state assignment should not change under tiny coordinate perturbations. Coordinate-level augmentation trains the encoder to ignore insignificant structural noise.

### Example

```python
import numpy as np


def jitter_coords(coords: np.ndarray, valid_mask: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    out = coords.copy()
    noise = rng.normal(0.0, std, size=out[valid_mask].shape).astype(out.dtype)
    out[valid_mask] += noise
    return out
```

---

## 12. Generate a preprocessing report for every dataset build

### Actionable steps

Write both `training_data_report.json` and `training_data_report.md` containing:

```text
number of input structures/domains
number rejected by each structure-level filter
number of alignments
number rejected by each alignment-level filter
number of residue pairs before filtering
number after descriptor-validity filtering
number after local distance filtering
number after capping/subsampling
feature means/stds/min/max/quantiles
NaN and Inf counts
Cα-distance quantiles
sequence-separation/contact-bin histogram
examples per group histogram
examples per source corpus histogram
random seed
preprocessing config checksum
```

### Reasoning

Model performance cannot be interpreted without knowing what the training data actually contains. Reports also make preprocessing changes auditable.

### Example report skeleton

```python
import json
from pathlib import Path


def write_report(report: dict, out_path: str) -> None:
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True))
```

---

## 13. Use deterministic preprocessing

### Actionable steps

1. Use a single explicit seed for all random subsampling.
2. Use `numpy.random.default_rng(seed)`.
3. Save the seed in the report and config.
4. Sort input manifests and alignment rows before processing.
5. Avoid filesystem-order-dependent behavior.

### Reasoning

Training already has stochasticity. Preprocessing should be reproducible so that model differences can be attributed to the model or training setup, not hidden data-order changes.

---

## 14. Use memory-mapped or chunked datasets when the corpus grows

### Actionable steps

1. For small datasets, in-memory tensors are acceptable.
2. For larger datasets, use memory-mapped NumPy arrays, Zarr, HDF5, Arrow, or Parquet-backed datasets.
3. Implement a PyTorch `Dataset` that reads rows lazily.
4. Use `DataLoader` parameters such as `num_workers`, `pin_memory`, and `persistent_workers` after benchmarking.

### Reasoning

Large structural corpora can exceed RAM. Lazy loading and chunking prevent preprocessing and training from becoming memory-bound.

### PyTorch reference pattern

PyTorch `DataLoader` supports map-style and iterable-style datasets, custom sampling order, automatic batching, multiprocessing, and memory pinning.

```python
import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, path: str):
        self.arr = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, idx: int):
        xy = self.arr[idx]
        x = torch.from_numpy(np.asarray(xy[:, 0], dtype=np.float32))
        y = torch.from_numpy(np.asarray(xy[:, 1], dtype=np.float32))
        return x, y
```

---

## 15. Keep validation data unaugmented, uncapped where feasible, and separately reported

### Actionable steps

1. Apply strict quality filters to validation data.
2. Do not apply coordinate noise to validation data.
3. Avoid aggressive residue-pair caps in validation unless necessary for runtime.
4. Report validation data statistics separately from training data.
5. Compute alphabet quality metrics on validation, not only reconstruction loss.

### Reasoning

Validation should measure the natural distribution of clean held-out data. Training-specific balancing and augmentation should not obscure whether the alphabet generalizes.

Recommended validation metrics:

```text
partner-prediction loss
state usage and perplexity
minimum state frequency
aligned-state mutual information
transition-adjusted mutual information
substitution-matrix quality
state stability under small perturbation
```

---

## 16. Preserve compact alignments but do not rely on them alone

### Actionable steps

1. Keep CIGAR or equivalent compact alignment strings in the alignment manifest.
2. Parse them into explicit aligned residue indices during preprocessing.
3. Store the expanded residue-pair table.
4. Validate that compact and expanded forms round-trip.

### Reasoning

Compact alignment strings are efficient and reproducible, but they do not contain enough information for training diagnostics. Expanded residue-pair rows are needed for filtering, weighting, and debugging.

### Example round-trip invariant

```text
parse(cigar) -> aligned index pairs -> regenerate compact alignment path -> same residue-pair set
```

---

## 17. Export data artifacts with model artifacts

### Actionable steps

When saving a trained alphabet, also save:

```text
preprocessing_config.json
feature_scaler.npz or feature_scaler.json
split_manifest.csv
training_data_report.json
validation_data_report.json
state_frequency_validation.csv
```

### Reasoning

A discrete alphabet cannot be evaluated independently of its feature definitions, scaling, filters, and training distribution. These artifacts are required for reproducibility and comparison across runs.

PyTorch recommends saving model `state_dict` objects as modular Python dictionaries of parameters. Apply the same principle to preprocessing: save explicit, inspectable artifacts rather than implicit code state.

---

# Minimal mandatory preprocessing pipeline

Implement these steps first:

```text
1. Create a split manifest before residue-pair generation.
2. Parse alignments into an expanded residue-pair table.
3. Filter by descriptor validity and local structural consistency.
4. Save provenance and quality metadata for every retained example.
5. Fit feature standardization on training data only.
6. Export the scaler and preprocessing config.
7. Weight or cap examples to prevent alignment/group domination.
8. Write JSON/Markdown reports for train and validation data.
9. Train using PyTorch Dataset/DataLoader abstractions rather than hard-coded full-array loading when scaling.
10. Evaluate validation state usage and aligned-state statistics, not only loss.
```

---

# External references

- PyTorch `torch.utils.data.DataLoader`, datasets, samplers, multiprocessing, and memory pinning: https://docs.pytorch.org/docs/2.12/data.html
- PyTorch data-loading optimization tutorial: https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html
- scikit-learn `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn preprocessing and train-set scaling guidance: https://scikit-learn.org/stable/modules/preprocessing.html
- PyTorch saving/loading models and `state_dict`: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
