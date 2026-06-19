# VQ-VAE v2 Configuration and CLI Guide

This guide details all configuration options, features, and command-line flags available in the modernized VQ-VAE (v2) model training pipeline.

The training pipeline uses a two-tier configuration structure:
1. **Tier 1 (Baseline Configuration):** Loaded from a YAML file containing all default configuration values (e.g. `configs/train/scop_vq_baseline.yaml`), structured under nested sections: `model`, `loss`, `data`, `optimizer`, and `training`.
2. **Tier 2 (Experimental Switch Overrides):** Any nested config value can be overridden via CLI flags at runtime.

---

## Command Usage

### Training Baseline Model
To train the baseline model using a baseline YAML configuration:
```bash
python scripts/train_v2.py --config configs/train/scop_vq_baseline.yaml
```

### Running with Experimental Flags
To run experimental configurations, append the override flags to the training command:
```bash
python scripts/train_v2.py \
  --config configs/train/scop_vq_baseline.yaml \
  --kmeans-init \
  --continuous-warmup-epochs 2 \
  --contrastive-weight 0.05
```

---

## Configuration Keys and CLI Flags

| Section | Key | CLI Override Flag | Description |
| :--- | :--- | :--- | :--- |
| **model** | `input_dim` | (None) | Dimension of input features (default: `10`). |
| | `hidden_dim` | (None) | Width of hidden layers (default: `64`). |
| | `z_dim` | (None) | Dimension of codebook vector space (default: `4`). |
| | `n_states` | `--n-states <int>` | Size of the discrete 3Di state representation alphabet (default: `20`). |
| | `norm` | (None) | Normalization layer used (default: `layernorm`). |
| | `activation` | (None) | Activation function used (default: `silu`). |
| | `quantizer` | `--quantizer-type <vq/fsq>` | Vector Quantizer backend: `vq` (EMA-VQ) or `fsq` (Finite Scalar Quantizer). |
| | `quantizer_gradient`| (None) | Gradient propagation estimator (default: `ste`). |
| | `kmeans_init` | `--kmeans-init` / `--no-kmeans-init` | Enable/disable seeding VQ codebook from data latents via k-means. |
| | `fsq_levels` | `--fsq-levels <int> [<int> ...]` | Explicit grid resolution steps for FSQ (e.g., `5 4`). |
| | `decay` | `--decay <float>` | EMA codebook moving average update decay coefficient. |
| | `eps` | `--eps <float>` | Laplace smoothing epsilon value. |
| | `l2_normalize` | `--l2-normalize` / `--no-l2-normalize` | Use L2 normalization/cosine distance for VQ codebook lookups. |
| | `min_count` | `--min-count <float>` | Count threshold below which a codebook code is considered dead. |
| | `replacement_warmup_steps`| `--replacement-warmup-steps <int>` | Wait steps before dead codebook replacement begins. |
| **loss** | `primary` | `--loss-type <smooth_l1/gaussian_nll>` | Reconstruction loss type choice: `smooth_l1` or `gaussian_nll`. |
| | `commitment_weight` | `--commitment-cost <float>` | Weighting coefficient for the quantization commitment loss. |
| | `usage_weight` | `--usage-weight <float>` | Multiplier for the code usage entropy regularization. |
| | `contrastive_weight` | `--contrastive-weight <float>` | Multiplier for the aligned-pair contrastive InfoNCE objective. |
| | `self_reconstruction_weight`| `--self-weight <float>` | Multiplier weighting self-reconstruction vs partner prediction. |
| **data** | `train_dir` | `--train-dir <path>` | Directory containing training data arrays. |
| | `val_dir` | `--val-dir <path>` | Directory containing validation data arrays. |
| | `max_ca_dist` | (None) | C-alpha distance cutoff filter for aligned pairs (default: `5.0`). |
| | `standardize_features`| (None) | Enable/disable train-only feature scaling standardizer. |
| | `sampler` | (None) | Random or alignment-aware sampler selection. |
| | `max_pairs_per_alignment`| (None) | Capping threshold constraint on examples per alignment. |
| | `descriptor_jitter_std`| `--descriptor-jitter-std <float>` | Standard deviation of descriptor-space feature jittering. |
| | `alignments_per_batch`| `--alignments-per-batch <int>` | Number of distinct alignments per batch in alignment-aware sampler. |
| **optimizer** | `type` | (None) | Optimizer type (default: `adamw`). |
| | `lr` | `--lr <float>` | Optimizer learning rate. |
| | `weight_decay` | `--weight-decay <float>` | Optimization weight decay multiplier. |
| | `exclude_bias_and_norm_from_decay`| (None) | Exclude norm/bias parameters from decay regularization (default: `true`). |
| | `gradient_clip_val` | (None) | Gradient clipping threshold (default: `1.0`). |
| | `warmup_steps` | (None) | Linear learning rate warmup step count. |
| | `warmup_ratio` | `--warmup-ratio <float>` | Warmup step ratio relative to total training steps. |
| | `scheduler` | (None) | Linear warmup + cosine decay scheduler selection. |
| **training** | `seed` | `--seed <int>` | Random seed for replication and training. |
| | `out_dir` | `--out-dir <path>` | Output path where checkpoints and scalers are saved. |
| | `max_epochs` | `--max-epochs <int>` | Maximum number of training epochs to fit. |
| | `batch_size` | `--batch-size <int>` | Batch size for training loaders. |
| | `continuous_warmup_epochs`| `--continuous-warmup-epochs <int>` | Epochs using continuous latents before quantization starts. |
| | `aux_ramp_epochs` | `--aux-ramp-epochs <int>` | Epochs to ramp auxiliary losses from 0 to 1. |
| | `accumulate_grad_batches`| `--accumulate-grad-batches <int>` | Gradient accumulation steps. |
| | `precision` | `--precision <str>` | Trainer precision (e.g., `32-true`, `bf16-mixed`). |
| | `torch_compile` | `--torch-compile` / `--no-torch-compile` | Enable/disable compiling the model via `torch.compile`. |
