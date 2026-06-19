# VQ-VAE v2 Configuration and CLI Guide

This guide details all configuration options, features, and command-line overrides available in the modernized VQ-VAE (v2) model training pipeline.

The training pipeline uses a two-tier configuration structure:
1. **Tier 1 (YAML Configuration):** Loaded from a YAML file containing nested configuration sections matching the `TrainConfig` schema (e.g. `configs/train/scop_v2_default.yaml`).
2. **Tier 2 (Dotted overrides):** Any nested config value can be overridden via command-line options using standard dotted notation, e.g., `--section.key value`.

---

## Command Usage

### Training Default Model
To train the model using the default YAML configuration:
```bash
python -m tdi.v2 train --config configs/train/scop_v2_default.yaml
```

### Running with Dotted CLI Overrides
To override specific parameters at runtime, append dotted flags:
```bash
python -m tdi.v2 train \
  --config configs/train/scop_v2_default.yaml \
  --quantizer.gradient_mode ste \
  --training.max_epochs 10 \
  --optimizer.lr 0.002
```

---

## Configuration Keys and Sections

Below are the sections and keys of the `TrainConfig` schema, mapped to the dotted CLI overrides.

### 1. `model`
Configuration parameters for the encoder/decoder network architecture.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `input_dim` | `--model.input_dim <int>` | `10` | Dimension of input features. |
| `hidden_dim` | `--model.hidden_dim <int>` | `64` | Width of residual MLP hidden layers. |
| `z_dim` | `--model.z_dim <int>` | `4` | Dimension of codebook vector space. |
| `n_states` | `--model.n_states <int>` | `20` | Size of the discrete 3Di state representation alphabet. |
| `quantizer_type`| `--model.quantizer_type <str>`| `"ema_vq"` | Quantizer backend choice: `"ema_vq"` or `"fsq"`. |
| `loss_type` | `--model.loss_type <str>` | `"smooth_l1"` | Choice of reconstruction loss (`"smooth_l1"` or `"gaussian_nll"`). |
| `fsq_levels` | `--model.fsq_levels <list[int]>`| `None` | Explicit grid resolution steps for FSQ (e.g., `[5, 4]`). |

### 2. `quantizer`
Parameters specific to VQ/FSQ codebook quantization and updates.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `l2_normalize` | `--quantizer.l2_normalize <bool>`| `True` | Use cosine distance / L2 normalization for VQ lookups. |
| `decay` | `--quantizer.decay <float>` | `0.99` | EMA decay rate for moving average statistics. |
| `commitment_cost`| `--quantizer.commitment_cost <float>`| `0.25` | Commiment penalty loss weighting factor. |
| `min_count` | `--quantizer.min_count <float>` | `1.0` | Minimum usage count below which codebook centroids are replaced. |
| `replacement_warmup_steps`| `--quantizer.replacement_warmup_steps <int>`| `500` | Step warmup count before codebook replacements start. |
| `gradient_mode` | `--quantizer.gradient_mode <str>`| `"rotation_trick"`| Gradient path mode: `"rotation_trick"` (Householder) or `"ste"`. |
| `kmeans_init` | `--quantizer.kmeans_init <bool>` | `True` | Seed VQ codebook centroids from warmed-up latents using k-means. |
| `kmeans_seed` | `--quantizer.kmeans_seed <int>` | `0` | Random seed for k-means fitting. |
| `kmeans_init_batches`| `--quantizer.kmeans_init_batches <int>`| `16` | Dataloader batches used for k-means fitting. |
| `kmeans_init_samples`| `--quantizer.kmeans_init_samples <int>`| `50000` | Maximum samples to collect for k-means fitting. |

### 3. `loss`
Multipliers for primary and auxiliary loss objectives.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `lambda_self` | `--loss.lambda_self <float>` | `0.05` | Weighting multiplier for self-reconstruction. |
| `lambda_usage` | `--loss.lambda_usage <float>` | `0.001` | Weighting multiplier for code usage entropy loss. |
| `lambda_contrast`| `--loss.lambda_contrast <float>`| `0.02` | Weighting multiplier for the aligned-pair contrastive loss. |
| `temperature` | `--loss.temperature <float>` | `0.1` | Initial temperature for contrastive logits. |

### 4. `training`
PyTorch Lightning training loop settings.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `batch_size` | `--training.batch_size <int>` | `512` | Size of mini-batches. |
| `max_epochs` | `--training.max_epochs <int>` | `20` | Maximum epochs to run. |
| `quantizer_warmup_epochs`| `--training.quantizer_warmup_epochs <int>`| `1` | Warmup epochs (bypassing quantizer) before discretization. |
| `aux_ramp_epochs`| `--training.aux_ramp_epochs <int>`| `1` | Epochs over which auxiliary losses ramp up from 0 to 1. |
| `precision` | `--training.precision <str>` | `"32-true"` | Trainer arithmetic precision. |
| `seed` | `--training.seed <int>` | `1` | Random seed for data loaders and model initialization. |
| `accumulate_grad_batches`| `--training.accumulate_grad_batches <int>`| `4` | Gradient accumulation steps. |

### 5. `optimizer`
Learning rate, scheduling, and regularizations.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `lr` | `--optimizer.lr <float>` | `0.001` | Base optimizer learning rate. |
| `weight_decay` | `--optimizer.weight_decay <float>`| `0.0001` | Weight decay rate for parameters with ndim >= 2. |
| `warmup_ratio` | `--optimizer.warmup_ratio <float>`| `0.03` | Linear learning rate warmup step ratio. |
| `gradient_clip_val`| `--optimizer.gradient_clip_val <float>`| `1.0` | Maximum gradient norm clip limit. |

### 6. `data`
Data loading paths and samplers.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `processed_dir` | `--data.processed_dir <path>` | `"data/processed/scop_ca5_v1"`| Directory containing preprocessed raw/scaler arrays. |
| `descriptor_jitter_std`| `--data.descriptor_jitter_std <float>`| `0.0` | Jitter std applied to descriptor inputs. |
| `sampler` | `--data.sampler <str>` | `"alignment_balanced"` | Data sampler choice: `"alignment_balanced"` or `"random"`. |
| `alignments_per_batch`| `--data.alignments_per_batch <int>`| `64` | Alignments per batch in the balanced sampler. |

### 7. `outputs`
Output folder locations.

| Key | Dotted CLI Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `out_dir` | `--outputs.out_dir <path>` | `"outputs/models/scop_v2_default_seed1"` | Path to save exported model artifacts and logs. |
