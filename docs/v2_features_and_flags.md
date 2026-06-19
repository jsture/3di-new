# VQ-VAE v2 Configuration and CLI Guide

This guide details all configuration options, features, and command-line flags available in the modernized VQ-VAE (v2) model training pipeline.

The training pipeline uses a two-tier configuration structure:
1. **Tier 1 (Baseline Configuration):** Loaded from a YAML file containing all default configuration values (e.g. `configs/train/scop_vq_baseline.yaml`).
2. **Tier 2 (Experimental Switch Overrides):** Any config value can be overridden via CLI flags at runtime.

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

| YAML Config Key | CLI Option / Flags | Description |
| :--- | :--- | :--- |
| **Basic Parameters** | | |
| `seed` | `--seed <int>` | Random seed value for replication and dataset indexing. |
| `train_dir` | `--train-dir <path>` | Folder containing generated training arrays (`data.npy`). |
| `val_dir` | `--val-dir <path>` | Folder containing generated validation arrays (`data.npy`). |
| `out_dir` | `--out-dir <path>` | Output path where trained model checkpoints and scalers are exported. |
| `n_states` | `--n-states <int>` | Size of the discrete 3Di state representation alphabet (e.g., `20`). |
| `quantizer_type` | `--quantizer-type <vq/fsq>` | Vector Quantizer backend: `vq` (EMA-VQ) or `fsq` (Finite Scalar Quantizer). |
| `fsq_levels` | `--fsq-levels <int> [<int> ...]` | Resolution steps per dimension when `quantizer_type` is `fsq` (e.g., `5 4`). |
| `max_epochs` | `--max-epochs <int>` | Maximum epochs to run during Lightning Trainer fit. |
| `batch_size` | `--batch-size <int>` | Miniature batch size. |
| `lr` | `--lr <float>` | Starting learning rate. |
| **Quantizer & Regularizers** | | |
| `kmeans_init` | `--kmeans-init` / `--no-kmeans-init` | Enable/disable seeding codebook centroids from data latents via k-means. |
| `quantizer_warmup_epochs` | `--continuous-warmup-epochs <int>` | Epochs to train using continuous latents before quantization begins. |
| `aux_ramp_epochs` | `--aux-ramp-epochs <int>` | Epochs over which auxiliary loss weights ramp from 0 to 1 after quantization starts. |
| `lambda_contrast` | `--contrastive-weight <float>` | Multiplier for the aligned-pair contrastive objective loss. |
| `lambda_usage` | `--usage-weight <float>` | Multiplier for the code usage entropy regularization loss. |
| `lambda_self` | `--self-weight <float>` | Multiplier weighting the self-reconstruction objective loss relative to partner. |
| `descriptor_jitter_std` | `--descriptor-jitter-std <float>` | Standard deviation of experimental Gaussian jitter added to descriptor features. |
| **Batch Composition & Hardware** | | |
| `alignments_per_batch` | `--alignments-per-batch <int>` | Number of distinct alignments per batch in alignment-aware sampler (default: 0 / off). |
| `accumulate_grad_batches` | `--accumulate-grad-batches <int>` | Gradient accumulation steps. |
| `precision` | `--precision <str>` | Trainer precision (e.g., `32-true`, `bf16-mixed`). |
| `torch_compile` | `--torch-compile` / `--no-torch-compile` | Enable/disable compiling the PyTorch module using `torch.compile`. |
| **Advanced Quantization Details** | | |
| `decay` | `--decay <float>` | Moving average update decay coefficient for codebook updates (EMA). |
| `eps` | `--eps <float>` | Laplace smoothing epsilon value. |
| `commitment_cost` | `--commitment-cost <float>` | Commit cost weighting scaling factor in VQ commitment loss. |
| `l2_normalize` | `--l2-normalize` / `--no-l2-normalize` | Enable/disable cosine distance / L2 normalization for codebook lookups. |
| `min_count` | `--min-count <float>` | Usage count threshold below which a code is considered dead. |
| `replacement_warmup_steps` | `--replacement-warmup-steps <int>` | Step offset before dead code replacement is activated. |
| `weight_decay` | `--weight-decay <float>` | Weight decay regularization multiplier. |
| `warmup_ratio` | `--warmup-ratio <float>` | Fraction of steps spent linearly ramping learning rate during warmup. |
| `loss_type` | `--loss-type <smooth_l1/gaussian_nll>` | Reconstruction loss type choice. |
