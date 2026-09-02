# TDI v2 Dataset and Model Upgrade Plan

## Purpose

This plan turns the current dataset-independent preprocessing recommendations into a concrete implementation sequence for robust discrete structural-state learning.

The goal is to train a small, exportable structural alphabet model that produces discrete residue states that are:

- robust to small coordinate or descriptor perturbations;
- well-used, with minimal dead states;
- predictive of structurally aligned partner residues;
- useful for downstream state substitution statistics and alignment;
- reproducible across dataset builds, training runs, and export/load cycles.

## Overall judgment

The proposed preprocessing plan is sound. It correctly focuses on the major risks in this project:

1. leakage from residue-level splitting;
2. noisy aligned-residue supervision;
3. non-auditable `.npy`-only training data;
4. feature scale domination;
5. overrepresentation of large folds, long alignments, or duplicated groups;
6. invalid descriptors and hidden NaN/Inf propagation;
7. validation metrics that measure neural loss rather than alphabet quality.

The plan should be implemented with two explicit constraints:

1. C-alpha distance filtering must only be applied after structural superposition or when aligned coordinates are already in a shared frame.
2. Validation should be reported in two forms: raw held-out validation and balanced diagnostic validation.

---

# Phase 1: Make preprocessing auditable and leakage-safe

## 1. Create a split manifest before generating residue pairs

### Action

Create a manifest file before any residue-pair expansion.

Required columns:

```text
source_corpus
structure_id
domain_id
group_id
split
sequence_cluster_id
structure_cluster_id
fold_id
superfamily_id
```

Use the strongest available `group_id` in this order:

```text
fold/topology ID
superfamily/homology ID
structure cluster ID
sequence cluster ID
domain ID
structure ID
```

Generate train, validation, and test splits at the group level. Reject any alignment whose endpoints are assigned to different splits during split-specific preprocessing.

### Reasoning

Residue pairs are not independent samples. Random residue-pair splits leak highly similar local environments across train and validation data. Group-level splitting gives a more honest estimate of whether the alphabet generalizes.

### Implementation target

Add a command:

```bash
tdi-v2 make-splits \
  --structure-manifest structures.csv \
  --out split_manifest.csv \
  --group-key auto \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 123
```

### Acceptance checks

- No `group_id` appears in more than one split.
- No alignment crosses train/validation/test partitions.
- The split manifest checksum is written to the preprocessing report.

---

## 2. Expand compact alignments into a residue-pair table

### Action

Keep compact alignments such as CIGAR strings, but materialize an expanded table with one row per aligned residue pair.

Required columns:

```text
row_id
source_corpus
alignment_id
structure_id_1
structure_id_2
domain_id_1
domain_id_2
group_id_1
group_id_2
split
residue_index_1
residue_index_2
alignment_score
global_quality_score
local_quality_score
ca_pair_distance
ca_distance_status
valid_descriptor_1
valid_descriptor_2
filter_status
filter_reason
```

Store descriptors separately, in the same row order:

```text
pairs.parquet
x.npy
y.npy
```

For large corpora, use chunked storage:

```text
pairs.parquet
features.zarr
```

### Reasoning

A plain `.npy` array of `(x, y)` pairs is not enough to debug training failures. The expanded table allows you to identify which alignments, groups, structures, contact types, and filters produced the retained training examples.

### Implementation target

Add a command:

```bash
tdi-v2 build-pairs \
  --split-manifest split_manifest.csv \
  --alignment-manifest alignments.csv \
  --pdb-dir data/pdb \
  --split train \
  --out data/processed/train
```

### Acceptance checks

- `pairs.parquet` row count equals `x.npy.shape[0]` and `y.npy.shape[0]`.
- Each row has a stable `row_id`.
- Empty alignments produce zero rows without crashing.
- CIGAR parsing returns arrays of shape `(N, 2)`, including `(0, 2)` for empty alignments.

---

## 3. Apply local structural consistency filtering correctly

### Action

Apply the C-alpha distance filter only if the two structures are in the same superposed coordinate frame.

Use this rule:

```text
if aligned/superposed coordinates are available:
    compute ca_pair_distance
    retain ca_pair_distance <= 5.0 Å
else:
    ca_pair_distance = null
    ca_distance_status = "unavailable"
    do not silently apply the filter
```

Do not compare raw coordinates from independent PDB files unless an alignment transform has been applied.

### Reasoning

The C-alpha distance threshold is intended to remove residue pairs that are not local structural counterparts after structural alignment. Raw coordinates from unrelated PDB coordinate frames are arbitrary relative to each other, so filtering them directly discards valid pairs and keeps invalid ones unpredictably.

### Implementation target

Add one of these supported modes:

```text
--ca-filter-mode unavailable
--ca-filter-mode transformed-coordinates
--ca-filter-mode upstream-filtered
```

Behavior:

- `unavailable`: mark distance missing; do not filter by distance.
- `transformed-coordinates`: apply stored alignment transform and filter by distance.
- `upstream-filtered`: trust upstream aligner; write this assumption to the report.

### Acceptance checks

- The report states which mode was used.
- If `transformed-coordinates` is selected, every retained row has finite `ca_pair_distance`.
- If `unavailable` is selected, no raw-coordinate C-alpha filtering occurs.

---

## 4. Enforce descriptor validity after all structural filters

### Action

Filter residue pairs in this order:

```text
alignment membership
residue index in bounds
valid local coordinates for structure 1
valid local coordinates for structure 2
valid descriptor for residue 1
valid descriptor for residue 2
local structural consistency, if available
quality/confidence filters
```

Fail the preprocessing run if any retained feature contains NaN or Inf.

### Reasoning

Descriptor validity depends on local residue context. A residue can exist in an alignment but still be invalid for feature extraction because neighboring residues or coordinates are missing.

### Implementation target

Add validation utilities:

```python
import numpy as np


def assert_finite_features(x: np.ndarray, name: str) -> None:
    finite = np.isfinite(x)
    if not finite.all():
        bad = int(np.size(x) - finite.sum())
        raise ValueError(f"{name} contains {bad} non-finite values")
```

### Acceptance checks

- Retained `x.npy` and `y.npy` contain no NaN or Inf.
- The report gives rejection counts by filter stage.
- Invalid descriptor rows retain provenance in the rejected-row report or aggregate filter counts.

---

## 5. Fit feature standardization on training data only

### Action

Fit the feature scaler on the training split only.

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

Apply the same scaler to train, validation, test, and inference inputs.

Save:

```text
feature_scaler.json
feature_scaler.npz
```

### Reasoning

The descriptor combines cosines, distances, clipped sequence distance, and log sequence distance. These dimensions have different scales. Scaling on the full dataset leaks validation/test distribution information into training, so the scaler must be fit on the train split only.

### Acceptance checks

- `feature_scaler.json` includes `mean`, `std`, `eps`, and `fit_split="train"`.
- Validation/test preprocessing refuses to fit a new scaler.
- Near-zero variance features are reported.

---

# Phase 2: Control sampling and dataset composition

## 6. Add per-alignment caps

### Action

Cap the number of residue-pair examples contributed by each structure/domain alignment.

Default:

```text
max_pairs_per_alignment = 512
```

Use deterministic sampling:

```python
import numpy as np


def cap_indices(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))
```

### Reasoning

Very long alignments can dominate training. Per-alignment caps prevent a few large structures from controlling the learned state boundaries.

### Acceptance checks

- The report includes pre-cap and post-cap row counts per alignment.
- The cap seed is stored.
- Validation is either uncapped or has its cap reported separately.

---

## 7. Add group and alignment-aware sample weights

### Action

Compute sample weights before training.

Recommended multiplicative components:

```text
w_alignment = 1 / examples_from_alignment
w_group = 1 / examples_from_group
w_structure_pair = 1 / examples_from_structure_pair
```

Normalize weights to have mean 1.

Use PyTorch `WeightedRandomSampler`:

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

### Reasoning

Uniform residue-pair sampling overweights long proteins, dense alignments, and overrepresented folds. Weighted sampling lets you preserve examples while controlling their effective contribution.

### Acceptance checks

- The report includes min/max/quantiles of sample weights.
- The effective number of samples is reported.
- Training logs specify whether weighted sampling was enabled.

---

## 8. Report and optionally balance contact composition

### Action

For every source residue, store:

```text
nearest_neighbor_sequence_separation
nearest_neighbor_distance
contact_bin
```

Use bins:

```text
local:       |i - j| <= 1
near-local:  2 <= |i - j| <= 4
medium:      5 <= |i - j| <= 16
long-range: |i - j| > 16
```

Start by reporting the distribution. Add contact-bin sampling only if the distribution is strongly dominated by local or near-local contacts.

### Reasoning

The target alphabet should describe tertiary interactions. If training is dominated by sequence-local neighbors, states may become local-backbone letters rather than tertiary-interaction letters.

### Acceptance checks

- Training and validation reports include contact-bin histograms.
- If contact reweighting is enabled, both raw and reweighted contact distributions are reported.

---

## 9. Keep validation raw and add a balanced diagnostic validation set

### Action

Create two validation views:

```text
val_raw       = natural held-out distribution
val_balanced  = diagnostic group/contact-balanced subset
```

Use `val_raw` for final model selection. Use `val_balanced` to detect failure modes hidden by dominant groups.

### Reasoning

Validation should estimate real held-out performance, but a balanced diagnostic subset helps identify whether the model fails on rare groups or rare contact types.

### Acceptance checks

- All validation metrics are reported separately for `val_raw` and `val_balanced`.
- Model checkpointing uses the explicitly configured validation score.

---

# Phase 3: Make preprocessing reproducible

## 10. Save versioned preprocessing configuration

### Action

Write `preprocessing_config.json` with:

```text
feature_version
feature_sign_convention
alignment_parser_version
coordinate_parser_version
virtual_center_parameters
filters and thresholds
ca_filter_mode
standardizer path
split manifest path
source manifest checksum
alignment manifest checksum
random seed
software versions
creation timestamp
```

### Reasoning

The learned alphabet is only interpretable relative to the exact preprocessing procedure. If feature definitions, sign conventions, filters, or scalers change, centroids and substitution matrices are not directly comparable.

### Acceptance checks

- Every trained model directory contains the preprocessing config.
- The config is copied into the model export directory.
- The config checksum is included in the training logs.

---

## 11. Generate preprocessing reports

### Action

For every dataset build, write:

```text
training_data_report.json
training_data_report.md
validation_data_report.json
validation_data_report.md
```

Each report must include:

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
C-alpha distance quantiles, if available
sequence-separation/contact-bin histogram
examples per group histogram
examples per source corpus histogram
sample-weight quantiles
random seed
config checksum
```

### Reasoning

Training results cannot be interpreted without knowing what data the model saw. Reports make preprocessing changes auditable.

### Acceptance checks

- Reports are generated automatically by `build-pairs`.
- Reports are committed or archived with every model run.
- Any nonzero NaN/Inf count fails the build.

---

## 12. Add deterministic preprocessing checks

### Action

For a small fixture corpus, assert:

```text
same input + same config + same seed -> same pair table checksum
same input + same config + same seed -> same x/y checksum
same input + same config + same seed -> same scaler
```

Use `numpy.random.default_rng(seed)` for all random choices. Sort manifests and alignments before processing.

### Reasoning

Training is already stochastic. Preprocessing should be deterministic so model differences are attributable to model changes, not hidden data-order changes.

### Acceptance checks

- Add `tests/test_v2_preprocessing_determinism.py`.
- CI fails if fixture checksums change without an explicit fixture update.

---

# Phase 4: Upgrade training infrastructure

## 13. Use explicit PyTorch Dataset/DataLoader abstractions

### Action

Implement a map-style dataset for in-memory or memory-mapped descriptor arrays.

```python
import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, mean: np.ndarray, std: np.ndarray):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.as_tensor(np.asarray(self.x[idx]), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(self.y[idx]), dtype=torch.float32)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return x, y
```

Use `DataLoader` with `num_workers`, `pin_memory`, and `persistent_workers` only after benchmarking.

### Reasoning

The code should work for both small and large corpora. Lazy loading prevents large structure datasets from becoming RAM-bound. PyTorch `DataLoader` supports map-style datasets, custom sampling order, batching, multiprocessing, and memory pinning.

### Acceptance checks

- Small datasets can still load fully in memory.
- Large datasets can train with memory-mapped arrays.
- Training output records DataLoader settings.

---

## 14. Save models as state_dict plus config

### Action

Save model artifacts as explicit dictionaries, not pickled modules.

```python
import torch

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": model_config,
    "feature_mean": mean,
    "feature_std": std,
    "preprocessing_config": preprocessing_config,
}

torch.save(checkpoint, "model.pt")
```

Export directory:

```text
model.pt
model_config.json
feature_scaler.json
preprocessing_config.json
centroids.npy or fsq_levels.json
letters.txt
state_frequency_validation.csv
```

### Reasoning

Saving `state_dict` is more portable than pickling whole modules. It decouples the learned parameters from the exact Python object and class definition used during training.

### Acceptance checks

- Export/load round-trip gives identical state assignments on a fixed fixture batch.
- FSQ exports include `fsq_levels`.
- VQ exports include centroids or EMA embeddings.
- Feature scaler is loaded automatically during inference.

---

## 15. Add CI and smoke tests

### Action

Add CI commands:

```bash
ruff check .
ruff format --check .
pyright src/tdi
pytest
pytest --cov=tdi
```

Add a smoke test that:

1. builds a tiny pair dataset;
2. fits a scaler;
3. trains for 10 steps;
4. exports a model;
5. loads it;
6. confirms identical state assignments before and after export.

### Reasoning

The model code has enough moving parts that construction tests are insufficient. CI should catch broken gradients, broken quantizer dimensions, broken export/load paths, and nondeterministic preprocessing.

### Acceptance checks

- CI runs on every pull request.
- The smoke test completes in less than one minute on CPU.
- Export/load consistency is mandatory.

---

# Phase 5: Improve the model itself

## 16. Keep VQ and FSQ as mandatory baselines

### Action

Train and compare these models for every serious experiment:

```text
VQ baseline:
  ResidualMLP encoder
  L2-normalized EMA vector quantizer
  20 or 32 states
  z_dim = 4 or 8

FSQ baseline:
  ResidualMLP encoder
  levels = [5, 4] for 20 states
  levels = [4, 4, 2] for 32 states
```

### Reasoning

FSQ is a strong robustness baseline because it replaces learned vector codebooks with fixed scalar quantization levels. This removes codebook-collapse maintenance mechanisms such as EMA updates, dead-code replacement, and entropy regularization. Keep VQ only if it clearly improves alphabet quality.

### Acceptance checks

- Every experiment table includes both VQ and FSQ unless explicitly waived.
- FSQ output dimensionality equals `len(levels)` before encoder/decoder construction.
- VQ and FSQ use identical train/validation splits and metrics.

---

## 17. Add symmetric x-to-y and y-to-x training

### Action

Compute the training objective in both directions.

```python
loss_xy = model.loss(x, y)
loss_yx = model.loss(y, x)
loss = 0.5 * (loss_xy + loss_yx)
```

If the dataset already stores both directions, add a test that confirms this behavior and avoid accidental duplication.

### Reasoning

The final substitution statistics are symmetric. The neural training objective should not depend on arbitrary source/target order.

### Acceptance checks

- A unit test verifies that swapping `(x, y)` does not break training.
- Training logs specify whether bidirectional pairs are explicit in the data or implicit in the loss.

---

## 18. Add a state-transition prediction head

### Action

Predict the aligned partner's discrete state from the source quantized state.

```python
state_logits_y = transition_head(z_q_x)
state_y = model.encode_states(y).detach()
loss_state = F.cross_entropy(state_logits_y, state_y)
```

Use a small weight initially:

```text
lambda_state_transition = 0.05
```

### Reasoning

The decoder predicts continuous descriptors, but the final product is a substitution alphabet. A state-transition head directly trains the discrete states to be predictive of aligned partner states.

### Acceptance checks

- Validation reports transition accuracy and transition cross-entropy.
- The auxiliary loss can be disabled by config.
- The inference path does not require the transition head.

---

## 19. Add validation-time state-transition metrics

### Action

During validation, accumulate counts:

```text
counts[state_x, state_y]
```

Report:

```text
state entropy
state perplexity
minimum state frequency
dead-state fraction
joint entropy
aligned-state mutual information
transition-adjusted mutual information
```

### Reasoning

The model is only useful if the resulting state alphabet has good substitution statistics. Validation should directly evaluate state usage and aligned-state information, not only reconstruction loss.

### Acceptance checks

- Metrics are reported for `val_raw` and `val_balanced`.
- Checkpointing can monitor a composite alphabet score.
- A dead-state fraction above threshold fails or warns.

---

## 20. Add state-margin and perturbation-stability metrics

### Action

For VQ, compute the distance margin between the nearest and second-nearest codebook vectors:

```python
margin = d_second - d_first
```

Report:

```text
mean_state_margin
p05_state_margin
fraction_margin_below_threshold
```

For all quantizers, evaluate perturbation stability:

```python
@torch.no_grad()
def state_stability(model, x, sigma: float = 0.03):
    states = model.encode_states(x)
    noisy_states = model.encode_states(x + sigma * torch.randn_like(x))
    return (states == noisy_states).float().mean()
```

Use:

```text
sigma = 0.01, 0.03, 0.05, 0.10
```

### Reasoning

A robust structural alphabet should not change state under small descriptor noise. Low-margin assignments indicate unstable decision boundaries.

### Acceptance checks

- Stability metrics appear in validation logs.
- Low-margin state fractions are tracked across training.
- Model selection includes stability as a diagnostic, not necessarily as the primary objective.

---

## 21. Initialize VQ centroids from k-means

### Action

Before full VQ training:

1. Train or initialize the encoder for a short warmup.
2. Encode a random subset of training examples.
3. Fit k-means with `K = n_states` in latent space.
4. Initialize VQ embeddings from k-means centroids.
5. Continue normal EMA VQ training.

### Reasoning

Random codebook initialization can waste early training and increase dead-code risk. K-means initialization gives each centroid a reasonable starting region.

### Acceptance checks

- K-means initialization is optional by config.
- Initial state usage is reported before and after initialization.
- If k-means fails or produces empty clusters, training falls back to random initialization with a warning.

---

## 22. Add optional context-window encoding

### Action

Add a config flag for local sequence context.

Simple version:

```text
input = concat(descriptor[i-2], descriptor[i-1], descriptor[i], descriptor[i+1], descriptor[i+2])
```

Alternative version:

```python
nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
```

Keep the default as single-residue descriptors.

### Reasoning

Small local context may stabilize state assignments. However, too much context can increase dependency between neighboring letters, reducing information density. Make this an experiment, not the default.

### Acceptance checks

- Context-window models are evaluated against the single-residue baseline.
- Transition-adjusted MI is monitored to detect excessive local-sequence dependence.
- Inference supports both modes through explicit config.

---

## 23. Add confidence-aware loss weights

### Action

Allow per-example loss weights:

```python
loss = (weights * per_example_loss).sum() / weights.sum().clamp_min(1e-8)
```

Possible weight components:

```text
alignment quality
local structural consistency
predicted-structure confidence
valid local context confidence
source corpus reliability
```

### Reasoning

Not all aligned residue pairs are equally trustworthy. Hard filtering discards data; confidence-aware weighting preserves examples while reducing the impact of noisy supervision.

### Acceptance checks

- Weight components are stored in `pairs.parquet`.
- Weighted and unweighted validation losses are reported separately.
- Loss weights are disabled by default until metadata quality is verified.

---

## 24. Add optional rotation-trick VQ only after baselines are stable

### Action

Add this as an experimental quantizer option:

```yaml
quantizer:
  type: vq
  gradient_estimator: straight_through  # or rotation_trick
```

Do not enable it by default.

### Reasoning

The rotation trick may improve gradient propagation through VQ assignments, but it is a second-stage optimization. It should not be implemented before the basic VQ and FSQ baselines, preprocessing, and validation metrics are reliable.

### Acceptance checks

- The implementation is behind a config flag.
- Results are compared against straight-through VQ on the same splits.
- Exported inference is unchanged.

---

# Phase 6: Model selection and reporting

## 25. Select checkpoints by alphabet quality, not only loss

### Action

Use a composite validation score:

```text
val_score =
    + aligned_state_mi
    + transition_adjusted_mi
    + normalized_state_entropy_weight * normalized_state_entropy
    - dead_state_penalty * dead_state_fraction
    - instability_penalty * (1 - state_stability_sigma_003)
```

Keep reconstruction or partner-prediction loss as a diagnostic, not the only model-selection target.

### Reasoning

The neural model is a way to learn discrete states. The selected model should have good state usage, state stability, and aligned-state information, because these are closer to downstream alignment utility than decoder loss alone.

### Acceptance checks

- The checkpoint monitor is written in the run config.
- All score components are logged separately.
- The best checkpoint and final checkpoint are both preserved.

---

## 26. Produce a run report for every trained alphabet

### Action

Write:

```text
run_report.json
run_report.md
state_frequency_train.csv
state_frequency_val_raw.csv
state_frequency_val_balanced.csv
state_transition_counts_val_raw.npy
state_transition_counts_val_balanced.npy
substitution_matrix.txt
```

The report should include:

```text
model config
training config
preprocessing config checksum
data report checksum
train/validation losses
state usage metrics
state stability metrics
state transition metrics
substitution matrix statistics
export/load consistency result
```

### Reasoning

The alphabet should be evaluated as a reusable artifact. The report makes it possible to compare alphabets trained with different data, quantizers, or preprocessing settings.

### Acceptance checks

- Every training run produces a complete report.
- The report links to exact model and preprocessing artifacts.
- Missing report sections fail the run in CI or strict mode.

---

# Minimal mandatory implementation order

Implement the following in order:

```text
1. Split manifest before residue-pair generation.
2. Expanded residue-pair table with provenance.
3. Empty-safe CIGAR parsing.
4. Correct C-alpha distance filtering mode.
5. Descriptor validity and finite-value checks.
6. Train-only feature scaler.
7. Preprocessing config and reports.
8. PairDataset + DataLoader with optional weighted sampling.
9. State_dict-based export/load with scaler metadata.
10. VQ and FSQ baseline training.
11. State usage, MI, and stability validation metrics.
12. Symmetric x-to-y / y-to-x training.
13. State-transition auxiliary head.
14. K-means VQ initialization.
15. Optional context-window and rotation-trick experiments.
```

---

# External references

- PyTorch `DataLoader`, datasets, samplers, batching, multiprocessing, and memory pinning: https://docs.pytorch.org/docs/2.12/data.html
- PyTorch saving/loading models and `state_dict`: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `state_dict` recipe: https://docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
- scikit-learn common pitfalls, including data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- FSQ: Finite Scalar Quantization, VQ-VAE Made Simple, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html
