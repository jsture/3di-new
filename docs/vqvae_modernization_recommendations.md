# VQ-VAE Modernization Recommendations for Robust 3Di-Like Discrete States

## Scope

These recommendations target a minimal, modern PyTorch model whose endpoint is a robust discrete structural alphabet, not an exact reproduction of the Foldseek paper model.

The target behavior is:

- assign every valid residue to one discrete state;
- use the available states well, without codebook collapse;
- make state assignments stable under small descriptor perturbations;
- make states predictive of structurally aligned partner residues;
- produce exported inference artifacts that remain usable across PyTorch versions;
- select models by downstream alphabet quality, not only by reconstruction loss.

---

## 1. Fix the reconstruction loss interface before making architectural changes

### Recommendation

Call `torch.nn.GaussianNLLLoss` as:

```python
recon_loss = self.loss_fn(mu, feat_y, var)
```

not as:

```python
recon_loss = self.loss_fn(feat_y, mu, var)
```

### Reasoning

`GaussianNLLLoss` expects `input` to be the predicted expectation, `target` to be the sampled target, and `var` to be the predicted positive variance. Passing `mu` as the target prevents the decoder mean from receiving the intended reconstruction gradient.

This is not merely a paper-fidelity issue. It is a correctness issue for any model trained to predict aligned partner descriptors.

### PyTorch reference

- `torch.nn.GaussianNLLLoss(input, target, var)`
- Documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html

---

## 2. Replace BatchNorm with LayerNorm in the encoder and decoder

### Recommendation

Replace `nn.BatchNorm1d(hidden_dim)` with `nn.LayerNorm(hidden_dim)` in all MLP blocks.

Use:

```python
nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.SiLU(),
)
```

instead of:

```python
nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
)
```

### Reasoning

LayerNorm removes dependence on batch statistics. This directly improves robustness because the model no longer changes behavior between training and evaluation due to BatchNorm running means and variances. It also removes the need to fuse BatchNorm into Linear layers for export.

For this project, the input is tabular geometry, not image-like data. BatchNorm is not essential, and its batch-size sensitivity creates avoidable complexity.

### PyTorch reference

- `torch.nn.LayerNorm`
- Documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

---

## 3. Use SiLU or GELU instead of ReLU

### Recommendation

Replace `nn.ReLU()` with `nn.SiLU()` as the default activation.

```python
activation = nn.SiLU()
```

### Reasoning

SiLU is smooth and avoids hard zeroing of negative activations. This is useful for small descriptor MLPs because the model has limited capacity and should preserve graded geometric information. ReLU is still acceptable, but SiLU is a stronger modern default for small residual MLPs.

### PyTorch reference

- `torch.nn.SiLU`
- Documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html

---

## 4. Increase model width and latent dimensionality modestly

### Recommendation

Use this default architecture:

```python
input_dim = 10
hidden_dim = 64
z_dim = 4
n_states = 20  # or 32 for experiments
```

Do not keep `hidden_dim=input_dim` unless the goal is strict minimality.

### Reasoning

The original hidden width of 10 is extremely small. If exact reproduction is not required, a width of 64 gives the encoder enough capacity to learn nonlinear decision surfaces while still keeping the model tiny.

A 2D latent is easy to visualize but unnecessarily restrictive. A 4D latent gives the quantizer more geometry while preserving a compact discrete bottleneck. The final output remains one discrete state, so increasing `z_dim` does not complicate downstream use.

---

## 5. Use a residual MLP encoder

### Recommendation

Replace the plain sequential encoder with a residual MLP.

```python
import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(depth)
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)
```

Use it as:

```python
encoder = ResidualMLP(input_dim=10, hidden_dim=64, output_dim=4, depth=3)
```

### Reasoning

Residual connections make the MLP easier to optimize and reduce the risk that added depth degrades training. This gives the encoder better nonlinear decision boundaries while remaining simple and fast.

---

## 6. Stabilize the decoder variance head or remove it

### Recommendation A: keep Gaussian NLL, but stabilize variance

Use `softplus` plus a small floor:

```python
import torch.nn.functional as F

raw_var = self.var_head(h)
var = F.softplus(raw_var) + 1e-4
```

Do not use unconstrained exponentiation as the default:

```python
var = torch.exp(self.logvar(h))
```

### Recommendation B: use SmoothL1Loss as the robust default

Start with:

```python
loss_recon = F.smooth_l1_loss(mu, feat_y)
```

and add the variance head only after the simpler loss is working.

### Reasoning

`exp(logvar)` can produce extremely small or very large variances, which can destabilize training. `softplus + floor` guarantees positivity while reducing overflow risk.

For this task, the endpoint is a useful alphabet, not calibrated uncertainty. `SmoothL1Loss` is often a better first objective because it is stable and less sensitive to outlier descriptor pairs.

### PyTorch references

- `torch.nn.functional.softplus`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
- `torch.nn.SmoothL1Loss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

---

## 7. Standardize input and target descriptors explicitly

### Recommendation

Fit feature scaling statistics on the training set and apply them to both `x` and `y`.

```python
mean = features.mean(axis=0)
std = features.std(axis=0).clip(min=1e-6)

features_scaled = (features - mean) / std
targets_scaled = (targets - mean) / std
```

Save `mean` and `std` with the exported model.

### Reasoning

The descriptor contains cosines, Euclidean distance, clipped sequence distance, and log sequence distance. These dimensions have different natural scales. Without explicit standardization, large-scale features can dominate the loss and the latent geometry.

Scaling also makes hyperparameters more transferable across datasets and virtual-center choices.

### Export recommendation

Save:

```text
feature_scaler.json
encoder_state_dict.pt
model_config.json
centroids.npy
letters.txt
```

---

## 8. Use L2-normalized vector quantization

### Recommendation

Normalize encoder outputs and codebook vectors before nearest-neighbor assignment.

```python
import torch.nn.functional as F

z_lookup = F.normalize(z, dim=-1)
codebook = F.normalize(self.embedding.weight, dim=-1)

distances = (
    z_lookup.pow(2).sum(dim=1, keepdim=True)
    + codebook.pow(2).sum(dim=1)
    - 2.0 * z_lookup @ codebook.t()
)
indices = distances.argmin(dim=1)
```

### Reasoning

Raw Euclidean assignment lets latent vector norms influence state assignment. L2 normalization makes assignment depend on direction rather than magnitude. This usually improves code usage and makes centroids easier to interpret.

### Relevant reference

- Improved VQGAN / ViT-VQGAN work reports low-dimensional lookup spaces and L2-normalized codes as useful for improving codebook utilization: https://openreview.net/forum?id=pfNyExj7z2

---

## 9. Replace gradient-updated codebooks with EMA codebook updates

### Recommendation

Implement an EMA quantizer as the default VQ backend.

```python
class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_states: int, z_dim: int, decay: float = 0.99,
                 eps: float = 1e-5, commitment_cost: float = 0.25):
        super().__init__()
        self.n_states = n_states
        self.z_dim = z_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost

        embedding = torch.randn(n_states, z_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_states))
        self.register_buffer("ema_sum", embedding.clone())

    def forward(self, z: torch.Tensor):
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2.0 * z @ self.embedding.t()
        )
        indices = distances.argmin(dim=1)
        encodings = F.one_hot(indices, self.n_states).type_as(z)
        z_q = encodings @ self.embedding

        if self.training:
            counts = encodings.sum(dim=0)
            sums = encodings.t() @ z.detach()

            self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
            self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)

            total = self.ema_count.sum()
            smoothed_count = (
                (self.ema_count + self.eps)
                / (total + self.n_states * self.eps)
                * total
            )
            self.embedding.copy_(self.ema_sum / smoothed_count.unsqueeze(1))

        commit_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()

        usage = encodings.float().mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

        return commit_loss, z_q, perplexity, indices, usage
```

### Reasoning

EMA codebook updates make centroids behave like online cluster means. This is a better default for a small structural alphabet because centroid locations are updated from assigned encoder outputs directly, rather than relying only on sparse codebook gradients.

The result is usually more stable codebook learning and easier monitoring of state usage.

### Relevant references

- Original VQ-VAE: https://arxiv.org/abs/1711.00937
- EMA VQ-VAE implementation pattern: https://github.com/zalandoresearch/pytorch-vq-vae

---

## 10. Add dead-code replacement

### Recommendation

Track moving state usage and replace unused codes with current encoder outputs from high-loss examples.

Implementation rule:

```python
unused = self.ema_count < min_count
```

For each unused code, sample a replacement vector from the current batch:

```python
replacement = z[high_loss_indices].detach()
self.embedding[unused] = replacement[:unused.sum()]
```

Use conservative thresholds:

```python
min_count = 1.0
replacement_warmup_steps = 500
```

### Reasoning

A dead code cannot become useful if it receives no assignments. Dead-code replacement gives the model a path to recover full alphabet usage. This matters because unused states directly reduce alphabet capacity.

Do not replace codes during the first few hundred steps; early usage is noisy.

---

## 11. Add weak usage regularization

### Recommendation

Add a small entropy term over batch-level code usage.

```python
usage = encodings.float().mean(dim=0)
entropy = -(usage * (usage + 1e-10).log()).sum()
loss = loss - lambda_usage * entropy
```

Start with:

```python
lambda_usage = 1e-3
```

### Reasoning

The alphabet should use most states. Weak entropy regularization discourages collapse while still allowing genuinely rare states. Strong entropy regularization is harmful because it can force biologically uncommon geometries to be overrepresented.

---

## 12. Add an FSQ baseline and keep it if it matches VQ performance

### Recommendation

Implement Finite Scalar Quantization as a competing backend.

For 20 states, use:

```python
levels = [5, 4]
```

For 32 states, use:

```python
levels = [4, 4, 2]
```

### Reasoning

FSQ replaces a learned vector codebook with scalar quantization over a few latent dimensions. It gives discrete states without learned codebook maintenance, commitment-loss tuning, EMA updates, or codebook reseeding.

Use FSQ if it gives comparable validation MI and ROC1. It is simpler and more robust by construction.

### Reference

- Finite Scalar Quantization: VQ-VAE Made Simple, ICLR 2024: https://arxiv.org/abs/2309.15505
- The paper states that FSQ avoids codebook collapse and does not require commitment losses, codebook reseeding, code splitting, or entropy penalties.

---

## 13. Train with partner prediction plus self-reconstruction

### Recommendation

Use a two-term decoder objective:

```python
loss_partner = F.smooth_l1_loss(mu_partner, feat_y)
loss_self = F.smooth_l1_loss(mu_self, feat_x)
loss = loss_partner + 0.1 * loss_self + vq_loss
```

### Reasoning

Partner prediction forces the state to encode conserved aligned geometry. Self-reconstruction prevents the state from discarding the local geometry of the source residue entirely. The self term should be weak because the main goal is conserved structural state learning, not ordinary autoencoding.

---

## 14. Add a contrastive aligned-partner objective

### Recommendation

Project quantized source states and target descriptors into a shared space and classify the correct aligned target within the batch.

```python
zq_x = source_projector(quantized_x)
h_y = target_projector(feat_y)

zq_x = F.normalize(zq_x, dim=-1)
h_y = F.normalize(h_y, dim=-1)

logits = zq_x @ h_y.T / temperature
labels = torch.arange(zq_x.shape[0], device=zq_x.device)
loss_contrast = F.cross_entropy(logits, labels)
```

Use:

```python
temperature = 0.1
lambda_contrast = 0.05
```

### Reasoning

Reconstruction losses can improve while downstream alphabet quality stagnates. A contrastive objective directly asks the discrete state to identify its aligned partner better than unrelated residues in the batch. This is closer to the downstream use case: finding structurally meaningful correspondences.

### PyTorch references

- `torch.nn.functional.normalize`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
- `torch.nn.functional.cross_entropy`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

---

## 15. Use AdamW and a cosine learning-rate schedule

### Recommendation

Use `torch.optim.AdamW` instead of Adam.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

Use cosine annealing when training for more than a few epochs:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs,
)
```

### Reasoning

AdamW decouples weight decay from momentum and variance accumulation. It is a better modern default when adding wider MLPs and residual blocks.

Cosine annealing gives stable late-stage convergence without manually choosing step drops.

### PyTorch references

- `torch.optim.AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- `torch.optim.lr_scheduler.CosineAnnealingLR`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

---

## 16. Add gradient clipping

### Recommendation

Use gradient norm clipping in the Lightning trainer.

```python
trainer = L.Trainer(
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)
```

### Reasoning

Quantized models can produce unstable gradients, especially when code assignments change rapidly early in training. Gradient clipping prevents rare batches from destabilizing the encoder.

### Lightning reference

- Gradient clipping in Lightning Trainer: https://lightning.ai/docs/pytorch/stable/common/optimization.html

---

## 17. Use early stopping and checkpoint the best validation model

### Recommendation

Add validation metrics and train with early stopping.

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_score", mode="max", patience=10),
    ModelCheckpoint(monitor="val_score", mode="max", save_top_k=1),
]

trainer = L.Trainer(
    max_epochs=100,
    callbacks=callbacks,
)
```

### Reasoning

A fixed four-epoch schedule is useful for reproducing a known pipeline but is not ideal for robust model development. Early stopping lets different model variants converge at their own rate and avoids selecting models based on arbitrary epoch counts.

The validation score should not be only reconstruction loss. Use a composite score based on downstream alphabet quality.

### Lightning references

- `EarlyStopping`: https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
- `ModelCheckpoint`: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html

---

## 18. Select models by alphabet metrics, not only by neural loss

### Recommendation

Track and use these validation metrics:

| Metric | Purpose |
|---|---|
| `val_partner_loss` | Confirms the model predicts aligned descriptors. |
| `state_perplexity` | Measures effective number of used states. |
| `min_state_frequency` | Detects dead or nearly dead states. |
| `state_entropy` | Measures usage balance. |
| `state_stability_noise` | Measures assignment robustness under descriptor perturbation. |
| `aligned_state_mi` | Measures information in aligned states. |
| `transition_adjusted_mi` | Penalizes trivial sequential dependence. |
| `roc1_family`, `roc1_superfamily`, `roc1_fold` | Measures downstream search utility. |

Define a validation score such as:

```python
val_score = (
    roc1_family
    + roc1_superfamily
    + roc1_fold
    + 0.05 * normalized_state_entropy
    - 0.10 * dead_state_fraction
)
```

### Reasoning

The model is only a means to create a structural alphabet. Reconstruction loss is a proxy. State usage, mutual information, substitution behavior, and search sensitivity are closer to the actual objective.

---

## 19. Add state-stability evaluation under perturbation

### Recommendation

Evaluate assignment stability after adding small standardized Gaussian noise.

```python
@torch.no_grad()
def state_stability(model, x, sigma: float = 0.03):
    states = model.encode_states(x)
    noisy_states = model.encode_states(x + sigma * torch.randn_like(x))
    return (states == noisy_states).float().mean()
```

Evaluate several noise levels:

```python
sigma_values = [0.01, 0.03, 0.05, 0.10]
```

### Reasoning

A useful structural alphabet should not change state assignments under tiny geometric noise. This metric detects overly sharp or unstable decision boundaries.

---

## 20. Add descriptor-level augmentation during training

### Recommendation

Train with small Gaussian noise in standardized descriptor space.

```python
if self.training:
    feat_x = feat_x + 0.02 * torch.randn_like(feat_x)
```

Optionally apply feature dropout:

```python
feat_x = F.dropout(feat_x, p=0.05, training=self.training)
```

### Reasoning

Small perturbations force the encoder to learn stable regions rather than brittle decision boundaries. This is especially useful when the endpoint is a discrete state assignment.

Do not use strong augmentation before confirming the baseline works.

---

## 21. Export `state_dict`, config, scaler, and centroids instead of pickled modules

### Recommendation

Save artifacts as:

```text
encoder_state_dict.pt
model_config.json
feature_scaler.json
centroids.npy
letters.txt
```

Save with:

```python
torch.save(encoder.state_dict(), out_dir / "encoder_state_dict.pt")
np.save(out_dir / "centroids.npy", centroids)
```

Load by reconstructing the model from `model_config.json`, then loading the state dict.

### Reasoning

Saving whole modules pickles Python objects and couples the artifact to the exact class definitions and environment. PyTorch recommends saving `state_dict` for flexibility when saving models for inference.

### PyTorch reference

- Saving and loading models: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

---

## 22. Keep the inference path simple and deterministic

### Recommendation

Provide one explicit inference method:

```python
@torch.no_grad()
def encode_states(self, x: torch.Tensor) -> torch.Tensor:
    self.eval()
    z = self.encoder(x)
    indices = self.quantizer.assign(z)
    return indices
```

The inference method should:

1. standardize features;
2. run the encoder;
3. assign nearest state;
4. return integer state IDs;
5. map IDs to letters outside the neural model.

### Reasoning

Training can have auxiliary heads and losses. Inference should be small, deterministic, and easy to test. Keep decoding, contrastive heads, and uncertainty heads out of the deployment path.

---

## 23. Use mixed precision only after correctness tests pass

### Recommendation

Enable mixed precision for training speed after validating numerics.

```python
trainer = L.Trainer(precision="16-mixed")
```

or, for manual PyTorch loops, use `torch.autocast` and `torch.amp.GradScaler`.

### Reasoning

Mixed precision can speed up training on suitable CUDA hardware, but quantizer distances, variance heads, and small numerical differences can affect assignments. Enable it only after matching full-precision validation metrics.

### PyTorch reference

- Automatic Mixed Precision: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

---

## 24. Use `torch.compile` only for the encoder after export tests pass

### Recommendation

Apply `torch.compile` only to the pure encoder/inference model, not initially to the training loop.

```python
compiled_encoder = torch.compile(encoder)
```

### Reasoning

The training path contains discrete assignment, EMA buffers, and non-standard update logic. Compile the simple inference path first. Verify that compiled and uncompiled state assignments are identical on a fixed validation set.

### PyTorch reference

- `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html

---

## 25. Add tests that check the objective, not just construction

### Recommendation

Add these tests.

#### Decoder mean receives gradients

```python
loss = lit_model.training_step((x, y), 0)
loss.backward()
assert model.decoder.mu.weight.grad is not None
assert model.decoder.mu.weight.grad.abs().sum() > 0
```

#### Quantizer returns all expected shapes

```python
vq_loss, z_q, perplexity, indices, usage = quantizer(z)
assert z_q.shape == z.shape
assert indices.shape == (z.shape[0],)
assert usage.shape == (n_states,)
```

#### Small model can overfit a tiny batch

```python
initial_loss = evaluate_loss(model, batch)
for _ in range(200):
    train_step(model, batch)
final_loss = evaluate_loss(model, batch)
assert final_loss < 0.5 * initial_loss
```

#### Exported and in-memory encoders agree

```python
states_before = model.encode_states(x)
load_exported_model(...)
states_after = exported_model.encode_states(x)
assert torch.equal(states_before, states_after)
```

### Reasoning

The current tests can pass even when the central objective is wrong. These tests catch training failures, quantizer shape errors, broken gradients, and export inconsistencies.

---

## Recommended default implementation

Use this as the first modernized model:

```text
Encoder: ResidualMLP(input_dim=10, hidden_dim=64, output_dim=4, depth=3)
Decoder: ResidualMLP-style decoder with mean head
Normalization: LayerNorm
Activation: SiLU
Quantizer: L2-normalized EMA VQ
Loss: SmoothL1 partner prediction + 0.1 self reconstruction + VQ commitment
Regularization: weak usage entropy, dead-code replacement after warmup
Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
Scheduler: CosineAnnealingLR
Training: early stopping on validation alphabet score
Export: state_dict + config + scaler + centroids
```

Then run this baseline:

```text
FSQ baseline: ResidualMLP + FSQ levels [5, 4] + same decoder/loss/evaluation
```

Keep the FSQ model if it matches VQ on downstream metrics. It is simpler and avoids codebook maintenance.

---

## Minimal migration sequence

1. Fix `GaussianNLLLoss(mu, target, var)` or switch to `SmoothL1Loss`.
2. Add descriptor standardization and export the scaler.
3. Replace BatchNorm/ReLU with LayerNorm/SiLU.
4. Replace the encoder with a residual MLP.
5. Increase `hidden_dim` to 64 and `z_dim` to 4.
6. Add L2-normalized EMA VQ.
7. Add usage metrics, dead-code replacement, and weak entropy regularization.
8. Add early stopping and checkpointing by validation alphabet score.
9. Add FSQ `[5, 4]` as a baseline.
10. Replace pickled module export with `state_dict + config + scaler + centroids`.

---

## Key references

- PyTorch `GaussianNLLLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
- PyTorch `LayerNorm`: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch `SiLU`: https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
- PyTorch `SmoothL1Loss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
- PyTorch `AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch model saving/loading: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- Lightning early stopping: https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
- Lightning checkpointing: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
- VQ-VAE: Neural Discrete Representation Learning: https://arxiv.org/abs/1711.00937
- FSQ: Finite Scalar Quantization, VQ-VAE Made Simple: https://arxiv.org/abs/2309.15505
- Improved VQGAN / codebook utilization discussion: https://openreview.net/forum?id=pfNyExj7z2
