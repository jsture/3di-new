# v2 VQ-VAE Implementation Action Plan

## Purpose

This plan turns the code review into concrete implementation tasks. The goal is to make the v2 VQ-VAE implementation reliable enough for benchmarking and downstream alphabet-quality evaluation.

The v2 code already contains many of the intended modernization features, including a residual encoder, LayerNorm/SiLU, EMA VQ, FSQ, SmoothL1/Gaussian losses, feature standardization, export helpers, and validation metrics. However, several parts are incomplete, incorrectly wired, or unsafe for reliable benchmarking.

Use this plan in order. The early items fix correctness issues; later items improve training quality and maintainability.

---

# Phase 1 — Fix correctness blockers

## 1. Fix FSQ dimensionality before constructing the model

### Problem

The FSQ path is likely broken when using the default settings.

The model currently constructs the encoder, decoder, and contrastive projector using the user-provided `z_dim`, then later replaces `self.z_dim` with `len(fsq_levels)` when `quantizer_type == "fsq"`.

With defaults:

```python
z_dim = 4
fsq_levels = [5, 4]
```

FSQ produces a quantized latent of shape `(batch, 2)`, but the decoder was built to expect `(batch, 4)`.

### Required change

Resolve the quantizer dimensionality before constructing the encoder, decoder, and projectors.

### Concrete implementation

In `TdiV2Model.__init__`, do this first:

```python
if quantizer_type == "fsq":
    fsq_levels = fsq_levels if fsq_levels is not None else [5, 4]
    z_dim = len(fsq_levels)
    n_states = int(np.prod(fsq_levels))
else:
    fsq_levels = None
```

Then construct model components:

```python
self.z_dim = z_dim
self.n_states = n_states
self.quantizer_type = quantizer_type
self.fsq_levels = fsq_levels

self.encoder = ResidualMLP(input_dim, hidden_dim, z_dim, depth=3)
self.decoder = Decoder(input_dim, hidden_dim, z_dim, loss_type=loss_type)
self.source_projector = nn.Linear(z_dim, contrastive_dim)
```

Then instantiate the quantizer:

```python
if quantizer_type == "fsq":
    self.quantizer = FSQQuantizer(fsq_levels)
elif quantizer_type == "ema_vq":
    self.quantizer = EMAVectorQuantizer(
        n_states=n_states,
        z_dim=z_dim,
        commitment_cost=commitment_cost,
    )
else:
    raise ValueError(f"Unknown quantizer_type: {quantizer_type}")
```

### Reasoning

All latent-space modules must agree on the latent dimension. FSQ defines its latent dimension from the number of scalar quantization levels. If the decoder and projectors are built with a different dimension, the model either fails at runtime or silently cannot be used as intended.

### Acceptance tests

Add tests:

```python
def test_fsq_default_forward_shapes():
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    x = torch.randn(8, 10)
    y = torch.randn(8, 10)
    out = model.training_step((x, y), 0)
    assert out.ndim == 0


def test_fsq_z_dim_matches_levels():
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    assert model.z_dim == 2
    assert model.encoder.output[-1].out_features == 2
```

---

## 2. Replace the current `forward()` method

### Problem

The current `forward()` returns discrete state IDs by calling `encode_states()`. That method uses `torch.no_grad()` and switches the model to eval mode.

This makes `forward()` unsuitable for training, Torch export, Lightning conventions, or normal PyTorch use.

### Required change

Make `forward()` a differentiable model pass. Keep discrete state encoding in a separate inference-only method.

### Concrete implementation

Replace:

```python
def forward(self, x):
    return self.encode_states(x)
```

with:

```python
def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    z = self.encoder(x)
    vq_loss, z_q, perplexity, indices, usage = self.quantizer(z)
    mu_partner, var_partner = self.decoder(z_q, partner=True)
    mu_self, var_self = self.decoder(z_q, partner=False)

    return {
        "z": z,
        "z_q": z_q,
        "indices": indices,
        "mu_partner": mu_partner,
        "var_partner": var_partner,
        "mu_self": mu_self,
        "var_self": var_self,
        "vq_loss": vq_loss,
        "perplexity": perplexity,
        "usage": usage,
    }
```

Update `training_step()` and `validation_step()` to use `self(x)` rather than manually duplicating the encode/quantize/decode sequence.

### Reasoning

`forward()` should be a normal differentiable computation. A model whose `forward()` disables gradients is hard to test, hard to export, and easy to misuse.

### Acceptance tests

```python
def test_forward_is_differentiable():
    model = TdiV2Model()
    x = torch.randn(4, 10, requires_grad=True)
    out = model(x)
    loss = out["mu_partner"].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
```

---

## 3. Remove mode mutation from `encode_states()`

### Problem

`encode_states()` currently calls:

```python
self.eval()
```

This silently changes the model state. A helper method should not mutate training/eval mode.

### Required change

Remove `self.eval()` from `encode_states()`.

### Concrete implementation

Use:

```python
@torch.no_grad()
def encode_states(self, x: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x)
    return self.quantizer.assign(z)
```

Then make callers explicitly set mode:

```python
model.eval()
states = model.encode_states(x)
```

### Reasoning

Implicitly switching to eval mode can corrupt training or validation control flow. In PyTorch, the caller should decide whether the model is in training or evaluation mode.

### Acceptance tests

```python
def test_encode_states_does_not_change_mode():
    model = TdiV2Model()
    model.train()
    x = torch.randn(4, 10)
    _ = model.encode_states(x)
    assert model.training is True
```

---

## 4. Fix the 5 Å Cα filtering logic

### Problem

The current implementation appears to compare Cα coordinates from two separate PDB structures directly:

```python
ca_dists = np.linalg.norm(ca1 - ca2, axis=1)
dist_mask = ca_dists <= max_ca_dist
```

This is only valid if the two structures have already been superposed into the same coordinate frame. If the pairfile only contains residue index mappings, this filter is invalid and will remove valid aligned residue pairs arbitrarily.

### Required change

Do not compare raw coordinates from independent PDB files unless a structural superposition transform has been applied.

Choose one of the following approaches.

### Option A — Preferred: apply TM-align transform before filtering

Store or parse the rotation and translation from the structural alignment step. Apply it to one structure before measuring Cα distances.

Implementation outline:

```python
coords2_aligned = coords2 @ rotation.T + translation
ca_dists = np.linalg.norm(coords1[idx_1, 0:3] - coords2_aligned[idx_2, 0:3], axis=1)
valid = ca_dists <= max_ca_dist
```

### Option B — Acceptable: move the filter upstream

If the TM-align/CIGAR-generation step already excludes residue pairs above 5 Å, remove the v2 raw-coordinate filter and document the upstream guarantee.

Use:

```python
if max_ca_dist is not None:
    raise NotImplementedError(
        "Cα distance filtering requires superposed coordinates or upstream filtered alignments."
    )
```

until the transform is available.

### Option C — Not acceptable

Do not keep the current raw-coordinate filter unless the input PDBs are explicitly guaranteed to share the same coordinate frame.

### Reasoning

Distance thresholds between aligned residues must be computed after structural superposition. Raw PDB coordinate frames are arbitrary relative to each other.

### Acceptance tests

Add one test using two identical coordinate sets where one is translated by a large vector. The filter should pass after applying a known inverse transform and fail without it.

```python
def test_ca_filter_requires_superposition():
    coords1 = np.array([[0.0, 0.0, 0.0]])
    coords2 = np.array([[100.0, 0.0, 0.0]])
    translation = np.array([-100.0, 0.0, 0.0])
    coords2_aligned = coords2 + translation
    assert np.linalg.norm(coords1[0] - coords2_aligned[0]) <= 5.0
```

---

## 5. Make CIGAR parsing empty-safe

### Problem

`align_features()` unpacks:

```python
idx_1, idx_2 = util.parse_cigar(cigar_string).T
```

If a CIGAR string has no matched residue pairs, parsing can return an empty 1D array. This causes unpacking errors.

### Required change

Make `parse_cigar()` always return an array with shape `(N, 2)`, including when `N == 0`.

### Concrete implementation

In `util.parse_cigar()`:

```python
return np.array(matches, dtype=np.int64).reshape(-1, 2)
```

Also guard in `align_features()`:

```python
idx_pairs = util.parse_cigar(cigar_string)
if idx_pairs.shape[0] == 0:
    return (
        np.empty((0, input_dim), dtype=np.float32),
        np.empty((0, input_dim), dtype=np.float32),
    )

idx_1, idx_2 = idx_pairs.T
```

If `input_dim` is not available before feature extraction, return after computing feature dimensions or use `10` for the current descriptor format.

### Reasoning

Data pipelines should handle empty or degenerate alignments without crashing the full preprocessing run.

### Acceptance tests

```python
def test_parse_cigar_empty_shape():
    pairs = parse_cigar("10M5I3D")
    assert pairs.shape == (0, 2)
```

---

# Phase 2 — Complete export and inference behavior

## 6. Save and load FSQ levels

### Problem

The export config stores `quantizer_type`, `z_dim`, and `n_states`, but not `fsq_levels`. This makes non-default FSQ models impossible to reconstruct reliably.

### Required change

Add `fsq_levels` to `model_config.json`.

### Concrete implementation

During export:

```python
config = {
    "input_dim": self.input_dim,
    "hidden_dim": self.hidden_dim,
    "z_dim": self.z_dim,
    "n_states": self.n_states,
    "quantizer_type": self.quantizer_type,
    "fsq_levels": self.fsq_levels,
    "loss_type": self.loss_type,
}
```

During load:

```python
model = cls(
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    z_dim=config["z_dim"],
    n_states=config["n_states"],
    quantizer_type=config["quantizer_type"],
    fsq_levels=config.get("fsq_levels"),
    loss_type=config.get("loss_type", "smooth_l1"),
)
```

### Reasoning

`n_states=20` does not uniquely define an FSQ quantizer. `[5, 4]`, `[2, 10]`, and `[20]` all produce 20 states but define different latent spaces and different integer encodings.

### Acceptance tests

```python
def test_fsq_export_roundtrip_preserves_levels(tmp_path):
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded = TdiV2Model.load_from_export(tmp_path)
    assert loaded.fsq_levels == [5, 4]
    assert loaded.z_dim == 2
```

---

## 7. Load and attach feature scaler metadata

### Problem

The export writes `feature_scaler.json`, but `load_from_export()` does not load or attach the scaler. The caller must manually remember to apply the scaler.

### Required change

Make exported models self-contained for inference.

### Concrete implementation

Option A: return model plus scaler.

```python
@classmethod
def load_from_export(cls, path: str | Path):
    ...
    with open(path / "feature_scaler.json") as f:
        scaler = json.load(f)
    return model, np.array(scaler["mean"]), np.array(scaler["std"])
```

Option B: attach scaler buffers to the model.

```python
model.register_buffer("feature_mean", torch.tensor(scaler["mean"], dtype=torch.float32))
model.register_buffer("feature_std", torch.tensor(scaler["std"], dtype=torch.float32))
```

Then add:

```python
@torch.no_grad()
def encode_scaled_states(self, x_raw: torch.Tensor) -> torch.Tensor:
    x = (x_raw - self.feature_mean) / self.feature_std
    return self.encode_states(x)
```

### Reasoning

State assignments depend on feature scaling. An exported encoder without the scaler is not a complete inference artifact.

### Acceptance tests

```python
def test_export_load_preserves_scaled_encoding(tmp_path):
    model = TdiV2Model()
    mean = np.random.randn(10)
    std = np.random.rand(10) + 0.1
    model.export_model(tmp_path, mean=mean, std=std)
    loaded, loaded_mean, loaded_std = TdiV2Model.load_from_export(tmp_path)
    assert np.allclose(mean, loaded_mean)
    assert np.allclose(std, loaded_std)
```

---

## 8. Export centroids consistently for both VQ and FSQ

### Problem

VQ models have centroids. FSQ models do not have a learned centroid table in the same sense. Export behavior should make this explicit.

### Required change

For VQ:

```text
centroids.npy
```

For FSQ:

```text
fsq_levels.json
```

Do not write fake centroids for FSQ unless downstream code explicitly requires representative latent points.

### Concrete implementation

```python
if self.quantizer_type == "ema_vq":
    np.save(out_dir / "centroids.npy", self.quantizer.embedding.detach().cpu().numpy())
elif self.quantizer_type == "fsq":
    with open(out_dir / "fsq_levels.json", "w") as f:
        json.dump({"levels": self.fsq_levels}, f)
```

### Reasoning

FSQ assignment is defined by scalar binning, not nearest learned centroids. Treating FSQ as if it had a VQ centroid file can create incorrect downstream inference assumptions.

### Acceptance tests

```python
def test_export_files_match_quantizer_type(tmp_path):
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    assert (tmp_path / "fsq_levels.json").exists()
    assert not (tmp_path / "centroids.npy").exists()
```

---

# Phase 3 — Complete training orchestration

## 9. Add a v2 training script or trainer factory

### Problem

The model logs `val_score`, and the optimizer uses AdamW plus a scheduler, but the shown code does not wire in early stopping, checkpointing, gradient clipping, train/validation splitting, or export from the best checkpoint.

### Required change

Create a v2 training entry point, for example:

```text
scripts/train_v2.py
```

or:

```text
src/tdi/v2/train.py
```

### Concrete implementation

The trainer should include:

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_score", mode="max", patience=10),
    ModelCheckpoint(monitor="val_score", mode="max", save_top_k=1),
]

trainer = L.Trainer(
    max_epochs=args.max_epochs,
    accelerator="auto",
    devices="auto",
    callbacks=callbacks,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)
```

After training, load the best checkpoint:

```python
best_path = checkpoint_callback.best_model_path
model = TdiV2Model.load_from_checkpoint(best_path)
model.export_model(out_dir, mean=scaler.mean, std=scaler.std)
```

### Reasoning

Model-level code is not enough. The recommendations depended on training-time behavior: early stopping, checkpointing by validation alphabet score, and gradient clipping. These must be wired into the actual executable training path.

### Acceptance tests

Add a smoke test that trains for two epochs on synthetic data and writes all expected export files.

```python
def test_train_v2_smoke(tmp_path):
    result = run_train_v2_synthetic(tmp_path, max_epochs=2)
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "feature_scaler.json").exists()
```

---

## 10. Define a real validation score for model selection

### Problem

The current `val_score` is useful but incomplete:

```text
-val_partner_loss + entropy bonus - dead-state penalty
```

This is not yet a direct downstream alphabet-quality score.

### Required change

Use a two-level validation setup:

1. Fast neural validation score during training.
2. Slower alphabet-quality benchmark for seed/model selection.

### Concrete implementation

During training, keep:

```python
val_score = -val_partner_loss + 0.05 * normalized_entropy - 0.10 * dead_state_fraction
```

After training each seed, compute:

```text
aligned_state_mi
transition_adjusted_mi
state_perplexity
min_state_frequency
noise_stability
ROC1 family/superfamily/fold if available
```

Then select by:

```python
selection_score = (
    roc1_family
    + roc1_superfamily
    + roc1_fold
    + 0.05 * normalized_entropy
    - 0.10 * dead_state_fraction
)
```

If ROC1 is not available during development, select by:

```python
selection_score = (
    aligned_state_mi
    + transition_adjusted_mi
    + 0.05 * normalized_entropy
    + 0.05 * noise_stability
    - 0.10 * dead_state_fraction
)
```

### Reasoning

Reconstruction loss is only a proxy. The purpose of the model is to generate a structural alphabet. Final model selection must include alphabet-level metrics.

### Acceptance tests

Create a validation report artifact:

```text
validation_metrics.json
```

with fields:

```json
{
  "val_partner_loss": 0.0,
  "state_perplexity": 0.0,
  "min_state_frequency": 0.0,
  "dead_state_fraction": 0.0,
  "noise_stability": 0.0,
  "aligned_state_mi": 0.0,
  "transition_adjusted_mi": 0.0,
  "selection_score": 0.0
}
```

---

# Phase 4 — Make the architecture match the intended modernization

## 11. Modernize the decoder trunk with residual blocks

### Problem

The encoder is a residual MLP, but the decoder is still a shallow non-residual MLP. This is not a correctness bug, but it means the architecture modernization is asymmetric.

### Required change

Use a residual decoder trunk.

### Concrete implementation

```python
class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, loss_type: str):
        super().__init__()
        self.loss_type = loss_type
        self.trunk = ResidualMLP(z_dim, hidden_dim, hidden_dim, depth=2)
        self.mu_partner = nn.Linear(hidden_dim, input_dim)
        self.mu_self = nn.Linear(hidden_dim, input_dim)

        if loss_type == "gaussian_nll":
            self.var_partner = nn.Linear(hidden_dim, input_dim)
            self.var_self = nn.Linear(hidden_dim, input_dim)
        else:
            self.var_partner = None
            self.var_self = None

    def forward(self, z_q: torch.Tensor, partner: bool = True):
        h = self.trunk(z_q)
        mu_head = self.mu_partner if partner else self.mu_self
        mu = mu_head(h)

        if self.loss_type != "gaussian_nll":
            return mu, None

        var_head = self.var_partner if partner else self.var_self
        var = F.softplus(var_head(h)) + 1e-4
        return mu, var
```

### Reasoning

A residual decoder makes optimization easier and makes the encoder/decoder design consistent. It is still small and fast.

### Acceptance tests

```python
def test_residual_decoder_outputs():
    decoder = Decoder(input_dim=10, hidden_dim=64, z_dim=4, loss_type="smooth_l1")
    z = torch.randn(8, 4)
    mu, var = decoder(z, partner=True)
    assert mu.shape == (8, 10)
    assert var is None
```

---

## 12. Make dead-code replacement deterministic under seed control

### Problem

Dead-code replacement is useful, but random replacement can make reproducibility weaker unless it is controlled by PyTorch's seeded RNG and logged.

### Required change

Use deterministic sampling from the current batch under the global PyTorch seed, and log replacements.

### Concrete implementation

```python
if self.training and self.steps > self.dead_code_warmup:
    dead = self.ema_count < self.dead_code_threshold
    n_dead = int(dead.sum().item())
    if n_dead > 0:
        perm = torch.randperm(z.shape[0], device=z.device)
        replacements = z.detach()[perm[:n_dead]]
        self.embedding[dead] = replacements
        self.ema_count[dead] = self.dead_code_threshold
```

Log:

```python
self.log("dead_codes_replaced", n_dead, prog_bar=False)
```

### Reasoning

Dead-code replacement should improve alphabet usage without making runs impossible to reproduce.

### Acceptance tests

```python
def test_dead_code_replacement_reproducible():
    torch.manual_seed(1)
    # run replacement once and save embeddings
    torch.manual_seed(1)
    # run again and compare
```

---

# Phase 5 — Improve data pipeline reliability

## 13. Make feature sign convention explicit

### Problem

The implementation uses `j - i` for sequence-distance features. The Foldseek paper describes `i - j`. Exact paper matching is not required, but the convention must be documented.

### Required change

Choose one convention and use it consistently.

### Recommended decision

Keep the current convention if existing data and models already depend on it, but rename variables to make it explicit:

```python
seq_delta = partner_idx - np.arange(len(partner_idx))
```

Then compute:

```python
seq_dist_clipped = np.clip(seq_delta, -4, 4)
seq_dist_log = np.sign(seq_delta) * np.log(np.abs(seq_delta) + 1)
```

Add a module-level comment:

```python
# Convention: sequence delta is partner_index - source_index.
# This is intentionally consistent with the existing implementation.
```

### Reasoning

Sign conventions are easy to accidentally flip between training and inference. A documented convention prevents silent incompatibilities.

### Acceptance tests

```python
def test_sequence_distance_convention():
    i = 10
    j = 14
    assert sequence_delta(i, j) == 4  # partner - source convention
```

---

## 14. Split preprocessing into explicit stages

### Problem

Training-data generation currently mixes feature extraction, alignment parsing, validity filtering, distance filtering, and bidirectional pair creation.

### Required change

Separate these stages:

1. `extract_features(pdb_path) -> features, mask, coords`
2. `parse_alignment(cigar) -> idx_1, idx_2`
3. `filter_valid_pairs(idx_1, idx_2, mask1, mask2)`
4. `filter_ca_distance(...)`
5. `make_bidirectional_pairs(feat1, feat2, idx_1, idx_2)`

### Reasoning

The Cα filter bug is easier to detect when each preprocessing step is explicit and tested independently.

### Acceptance tests

Each stage should have one unit test using synthetic inputs.

---

# Phase 6 — Add missing tests before benchmarking

## 15. Add objective-level tests

### Required tests

#### Decoder mean receives gradient

```python
def test_decoder_mean_receives_gradient():
    model = TdiV2Model(loss_type="smooth_l1")
    x = torch.randn(16, 10)
    y = torch.randn(16, 10)
    loss = model.training_step((x, y), 0)
    loss.backward()
    assert model.decoder.mu_partner.weight.grad is not None
    assert model.decoder.mu_partner.weight.grad.abs().sum() > 0
```

#### Tiny batch overfit

```python
def test_tiny_batch_overfit():
    model = TdiV2Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    initial = float(model.training_step((x, y), 0).detach())
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.training_step((x, y), 0)
        loss.backward()
        optimizer.step()
    final = float(model.training_step((x, y), 0).detach())

    assert final < initial
```

#### Exported and loaded model gives same states

```python
def test_export_roundtrip_states_identical(tmp_path):
    model = TdiV2Model()
    model.eval()
    x = torch.randn(16, 10)
    states_before = model.encode_states(x)

    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded, _, _ = TdiV2Model.load_from_export(tmp_path)
    loaded.eval()
    states_after = loaded.encode_states(x)

    assert torch.equal(states_before, states_after)
```

### Reasoning

Construction tests are insufficient. The code must prove that the objective trains, gradients flow, and exported inference reproduces in-memory inference.

---

# Phase 7 — Benchmark only after blockers are fixed

## 16. Run a three-model benchmark

### Required models

Train and compare:

| Model | Quantizer | Purpose |
|---|---|---|
| v1-compatible baseline | original VQ | Reference behavior |
| v2 VQ | EMA VQ, z_dim=4 | Main modernized model |
| v2 FSQ | FSQ `[5, 4]` | Robust no-codebook baseline |

### Required metrics

Report:

```text
partner prediction loss
state perplexity
state frequency histogram
min state frequency
dead state fraction
noise stability
aligned-state MI
transition-adjusted MI
ROC1 family
ROC1 superfamily
ROC1 fold
```

### Reasoning

The point of the modernization is not to reduce training loss. It is to produce better or more stable discrete states. FSQ should be kept if it performs similarly to VQ, because it removes codebook-maintenance complexity.

### Output artifact

Write:

```text
benchmark_report.md
benchmark_metrics.json
state_usage.tsv
```

---

# Recommended implementation order

Use this exact order:

1. Fix FSQ dimensionality.
2. Replace `forward()` with a differentiable forward pass.
3. Remove `self.eval()` from `encode_states()`.
4. Fix or disable the 5 Å Cα filter until superposed coordinates are available.
5. Make `parse_cigar()` empty-safe.
6. Save/load `fsq_levels`.
7. Load and attach or return feature scaler metadata.
8. Make FSQ export explicit and separate from VQ centroid export.
9. Add a v2 training script with early stopping, checkpointing, and gradient clipping.
10. Add objective-level tests.
11. Modernize the decoder trunk.
12. Document sequence-distance sign convention.
13. Refactor preprocessing into explicit stages.
14. Run VQ vs FSQ vs baseline benchmarks.

---

# Definition of done

The v2 implementation is ready for serious benchmarking only when all of the following are true:

- `quantizer_type="fsq"` works with default `fsq_levels=[5, 4]`.
- `forward()` is differentiable and does not call `encode_states()`.
- `encode_states()` does not mutate train/eval mode.
- Cα filtering is either correctly computed after superposition or explicitly disabled.
- Empty CIGAR alignments do not crash preprocessing.
- Export/import preserves model config, FSQ levels, scaler statistics, and state assignments.
- A v2 training script uses early stopping, checkpointing, and gradient clipping.
- Unit tests verify gradient flow, overfitting, quantizer shape correctness, and export consistency.
- Benchmark reports include state usage, stability, mutual information, and ROC1 metrics.

---

# Priority summary

## Must fix before any benchmark

1. FSQ dimension bug.
2. Invalid raw-coordinate Cα filtering.
3. Non-differentiable `forward()`.
4. `encode_states()` mutating eval mode.
5. Empty CIGAR crash.

## Should fix before model selection

1. Save/load `fsq_levels`.
2. Save/load scaler metadata cleanly.
3. Add training script with callbacks and gradient clipping.
4. Add objective-level tests.

## Nice to fix after correctness

1. Residual decoder trunk.
2. Deterministic dead-code replacement logging.
3. Refactored preprocessing stages.
4. More advanced selection score using ROC1 and mutual information.
