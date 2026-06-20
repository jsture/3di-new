"""VQ-VAE Model and Quantization Modules for v2.

This module implements:
- Residual MLP networks (encoder/decoder).
- Exponential Moving Average (EMA) Vector Quantizer with L2 normalization and dead code replacement.
- Finite Scalar Quantizer (FSQ) baseline.
- PyTorch Lightning Module (TdiV2Model) wrapping all optimization, losses, and logging.
"""

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.cluster import KMeans

from tdi.v2.quantizer_gradients import apply_quantizer_gradient


class ResidualMLP(nn.Module):
    """Residual Multi-Layer Perceptron (MLP) for v2.

    Uses LayerNorm and SiLU activations for robust optimization of structural features.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3) -> None:
        """Initialize the ResidualMLP.

        Args:
            input_dim: Number of input features.
            hidden_dim: Projection dimension inside blocks.
            output_dim: Dimension of output vectors.
            depth: Number of residual blocks.
        """
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResidualMLP.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Output tensor of shape (N, output_dim).
        """
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


def _kmeans(x: torch.Tensor, n_clusters: int, seed: int = 0) -> torch.Tensor:
    """Fit k-means centroids with a fixed seed.

    Args:
        x: Data points of shape (N, D).
        n_clusters: Number of centroids to fit.
        seed: Fixed seed for reproducible centroid initialization.

    Returns:
        Centroids of shape (n_clusters, D), on the same device/dtype as ``x``.
    """
    points = x.detach().cpu().numpy()
    # KMeans requires n_clusters <= n_samples; cap then pad the degenerate case below.
    k = min(n_clusters, points.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(points)
    centers = torch.tensor(kmeans.cluster_centers_, device=x.device, dtype=x.dtype)

    # Pad with random points if fewer samples than clusters (degenerate batch).
    if k < n_clusters:
        pad = x[torch.randint(x.shape[0], (n_clusters - k,), device=x.device)]
        centers = torch.cat([centers, pad], dim=0)
    return centers


def _quantizer_distances(
    z: torch.Tensor, codebook: torch.Tensor, l2_normalize: bool
) -> torch.Tensor:
    """Squared (cosine when ``l2_normalize``) distances between latents and codebook.

    Always computed in fp32 with autocast disabled so the deterministic inference path
    (``encode_states``) and the validation margin match the fp32 math used during EMA
    training, even under bf16 autocast.

    Args:
        z: Latent tensor of shape (N, z_dim).
        codebook: Codebook tensor of shape (K, z_dim).
        l2_normalize: If True, normalize both sides (cosine distance).

    Returns:
        Distance matrix of shape (N, K), in fp32.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        z32 = z.float()
        cb = codebook.float()
        if l2_normalize:
            z32 = F.normalize(z32, dim=-1)
            cb = F.normalize(cb, dim=-1)
        # Distance computation: d = x^2 + y^2 - 2xy
        return z32.pow(2).sum(dim=-1, keepdim=True) + cb.pow(2).sum(dim=-1) - 2.0 * z32 @ cb.t()


class EMAVectorQuantizer(nn.Module):
    """Vector Quantizer using Exponential Moving Average (EMA) codebook updates.

    Performs L2-normalized nearest neighbor search and implements dead-code replacement
    to avoid codebook collapse.
    """

    embedding: torch.Tensor
    ema_count: torch.Tensor
    ema_sum: torch.Tensor
    step_counter: torch.Tensor
    initialized: torch.Tensor

    def __init__(
        self,
        n_states: int,
        z_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        l2_normalize: bool = True,
        min_count: float = 1.0,
        replacement_warmup_steps: int = 500,
        gradient_mode: str = "rotation_trick",
    ) -> None:
        """Initialize the EMAVectorQuantizer.

        Args:
            n_states: Number of discrete states in the codebook.
            z_dim: Dimension of continuous latent space.
            decay: Exponential decay rate for moving average statistics.
            eps: Laplace smoothing epsilon.
            commitment_cost: Loss multiplier weighting the commitment penalty.
            l2_normalize: If True, uses cosine distance (L2 normalization) for lookups.
            min_count: Minimum EMA usage count threshold for code replacement.
            replacement_warmup_steps: Warmup step count before replacing unused centroids.
            gradient_mode: Gradient propagation mode ("ste" or "rotation_trick").
        """
        super().__init__()
        self.n_states = n_states
        self.z_dim = z_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.l2_normalize = l2_normalize
        self.min_count = min_count
        self.replacement_warmup_steps = replacement_warmup_steps
        self.gradient_mode = gradient_mode

        # Initialize codebook embedding weights uniformly
        embedding = torch.randn(n_states, z_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_states))
        self.register_buffer("ema_sum", embedding.clone())
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def init_codebook(self, z: torch.Tensor, seed: int = 0) -> None:
        """Seed the codebook from k-means of real encoder outputs.

        Replaces random init so early code usage is non-arbitrary. Runs k-means in the
        same space used for lookups (L2-normalized if ``self.l2_normalize``).

        Args:
            z: Encoder outputs of shape (N, z_dim).
            seed: Fixed seed for reproducible k-means centroid initialization.
        """
        features = F.normalize(z, dim=-1) if self.l2_normalize else z
        features = features.detach().to(self.embedding.dtype)

        centers = _kmeans(features, self.n_states, seed=seed)

        self.embedding.copy_(centers)
        self.ema_sum.copy_(centers)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform vector quantization.

        Args:
            z: Input continuous latents of shape (N, z_dim).

        Returns:
            Tuple of (commit_loss, z_q, perplexity, indices, usage, n_replaced).
        """
        # Distance lookup, one-hot, and EMA updates must stay fp32 (codebook math).
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            distances = _quantizer_distances(z32, self.embedding, self.l2_normalize)
            indices = distances.argmin(dim=-1)

            # Encodings matrix
            encodings = F.one_hot(indices, self.n_states).float()
            z_q = encodings @ self.embedding

            n_replaced = 0
            if self.training:
                self.step_counter += 1
                counts = encodings.sum(dim=0)
                sums = encodings.t() @ z32.detach()

                # Update moving averages
                self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)

                # Laplace smoothed count updates
                total = self.ema_count.sum()
                smoothed_count = (
                    (self.ema_count + self.eps) / (total + self.n_states * self.eps) * total
                )
                self.embedding.copy_(self.ema_sum / smoothed_count.unsqueeze(1))

                # Dead-code replacement
                if self.step_counter > self.replacement_warmup_steps:
                    dead = self.ema_count < self.min_count
                    n_dead = int(dead.sum().item())
                    if n_dead > 0:
                        perm = torch.randperm(z.size(0), device=z.device)
                        n_to_replace = min(n_dead, z.size(0))
                        dead_indices = torch.where(dead)[0][:n_to_replace]
                        replacements = z32.detach()[perm[:n_to_replace]]
                        self.embedding[dead_indices] = replacements
                        self.ema_count[dead_indices] = self.min_count
                        self.ema_sum[dead_indices] = replacements * self.min_count
                        n_replaced = n_to_replace

            usage = encodings.mean(dim=0)
            perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

        # Cast back to the working dtype for downstream (possibly bf16) ops.
        z_q = z_q.to(z.dtype)

        # Commitment loss to regularize encoder space
        commit_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        # Apply the selected surrogate-gradient estimator
        z_q = apply_quantizer_gradient(
            z=z,
            z_q=z_q,
            mode=self.gradient_mode,
            eps=self.eps,
        )

        return (
            commit_loss,
            z_q,
            perplexity,
            indices,
            usage,
            torch.tensor(n_replaced, device=z.device),
        )


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer (FSQ) baseline backend for v2.

    Replaces learned vector embeddings with fixed discrete scalar steps
    over the continuous latent space, avoiding codebook collapse.
    """

    basis: torch.Tensor
    implicit_codebook: torch.Tensor

    def __init__(
        self,
        levels: list[int],
        gradient_mode: str = "rotation_trick",
        eps: float = 1e-8,
    ) -> None:
        """Initialize the FSQQuantizer.

        Args:
            levels: Integer quantization steps for each dimension (e.g. [5, 4] for 20 states).
            gradient_mode: Gradient propagation mode ("ste" or "rotation_trick").
            eps: Epsilon value.
        """
        super().__init__()
        self.levels = levels
        self.n_states = int(np.prod(levels))
        self.z_dim = len(levels)
        self.gradient_mode = gradient_mode
        self.eps = eps

        # Coordinate basis coefficients
        basis: list[int] = []
        current = 1
        for level in reversed(levels):
            basis.append(current)
            current *= level
        self.register_buffer("basis", torch.tensor(list(reversed(basis)), dtype=torch.long))
        self.register_buffer("implicit_codebook", self._make_implicit_codebook())

    def _make_implicit_codebook(self) -> torch.Tensor:
        grids: list[torch.Tensor] = []
        for level in self.levels:
            if level % 2 == 1:
                grid = torch.linspace(-1.0, 1.0, level)
            else:
                grid = torch.linspace(-1.0 + 1.0 / level, 1.0 - 1.0 / level, level)
            grids.append(grid)
        mesh = torch.meshgrid(*grids, indexing="ij")
        codebook = torch.stack([m.flatten() for m in mesh], dim=1)
        return codebook

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Scale to range [-1, 1]
        z_bounded = torch.tanh(z)

        z_q_list: list[torch.Tensor] = []
        indices = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

        for i, level in enumerate(self.levels):
            if level % 2 == 1:
                val = (z_bounded[:, i] + 1.0) / 2.0 * (level - 1)
                val_q = torch.round(val)
                mapped = val_q / (level - 1) * 2.0 - 1.0
                idx = val_q.long()
            else:
                val = (z_bounded[:, i] + 1.0 - 1.0 / level) / 2.0 * (level - 1)
                val_q = torch.round(val)
                mapped = val_q / (level - 1) * (2.0 - 2.0 / level) - 1.0 + 1.0 / level
                idx = val_q.long()

            z_q_list.append(mapped)
            indices += idx * self.basis[i]

        z_q = torch.stack(z_q_list, dim=1)
        return z_q, indices

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, indices = self.quantize(z)
        z_q = apply_quantizer_gradient(
            z=z,
            z_q=z_q,
            mode=self.gradient_mode,
            eps=self.eps,
        )

        encodings = F.one_hot(indices, self.n_states).type_as(z)
        usage = encodings.float().mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

        # FSQ requires no commitment loss
        return (
            torch.tensor(0.0, device=z.device),
            z_q,
            perplexity,
            indices,
            usage,
            torch.tensor(0, device=z.device),
        )


class Decoder(nn.Module):
    """Decoder MLP reconstructing input/target features from quantized latents."""

    def __init__(
        self, input_dim: int, hidden_dim: int, z_dim: int, loss_type: str = "smooth_l1"
    ) -> None:
        """Initialize the Decoder.

        Args:
            input_dim: Feature dimension.
            hidden_dim: MLP projection width.
            z_dim: Latent representation width.
            loss_type: Underlying loss objective ("smooth_l1" or "gaussian_nll").
        """
        super().__init__()
        self.loss_type = loss_type
        self.trunk = ResidualMLP(z_dim, hidden_dim, hidden_dim, depth=2)
        self.mu_partner = nn.Linear(hidden_dim, input_dim)
        self.mu_self = nn.Linear(hidden_dim, input_dim)

        self.var_partner: nn.Linear | None
        self.var_self: nn.Linear | None
        if loss_type == "gaussian_nll":
            self.var_partner = nn.Linear(hidden_dim, input_dim)
            self.var_self = nn.Linear(hidden_dim, input_dim)
        else:
            self.var_partner = None
            self.var_self = None

    def forward(
        self, z_q: torch.Tensor, partner: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode the quantized representation.

        Args:
            z_q: Quantized latent vector.
            partner: If True, reconstructs the aligned partner; otherwise self.

        Returns:
            Tuple of (mu, var) reconstruction outputs.
        """
        h = self.trunk(z_q)
        mu_head = self.mu_partner if partner else self.mu_self
        mu = mu_head(h)

        if self.loss_type != "gaussian_nll":
            return mu, None

        var_head = self.var_partner if partner else self.var_self
        assert var_head is not None
        var = F.softplus(var_head(h)) + 1e-4
        return mu, var


class TdiV2Model(L.LightningModule):
    """Main modernized VQ-VAE model wrapping optimization, logging, and evaluation."""

    quantizer: EMAVectorQuantizer | FSQQuantizer
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        z_dim: int = 4,
        n_states: int = 20,
        quantizer_type: str = "vq",
        fsq_levels: list[int] | None = None,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        l2_normalize: bool = True,
        min_count: float = 1.0,
        replacement_warmup_steps: int = 500,
        lambda_usage: float = 1e-3,
        lambda_contrast: float = 0.02,
        lambda_self: float = 0.05,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.03,
        quantizer_warmup_epochs: int = 1,
        aux_ramp_epochs: int = 1,
        loss_type: str = "smooth_l1",
        kmeans_init: bool = True,
        kmeans_init_batches: int = 8,
        kmeans_seed: int = 0,
        kmeans_max_samples: int | None = None,
        gradient_mode: str = "rotation_trick",
    ) -> None:
        """Initialize the TdiV2Model.

        Args:
            input_dim: Dimension of input features.
            hidden_dim: Width of hidden layers.
            z_dim: Dimension of codebook vector space.
            n_states: Number of discrete states.
            quantizer_type: "vq" (EMA vector quantization) or "fsq" (finite scalar quantization).
            fsq_levels: Explicit grid resolution limits for FSQ.
            decay: Quantization moving average decay coefficient.
            eps: Laplace smoothing epsilon.
            commitment_cost: Commitment penalty multiplier.
            l2_normalize: Whether to normalize representations.
            min_count: Count threshold for dead codebook replacement.
            replacement_warmup_steps: Step warmup constraint for dead codebook replacement.
            lambda_usage: Code usage entropy weight.
            lambda_contrast: Aligned contrastive objective weight.
            lambda_self: Self-reconstruction objective weight.
            temperature: Softmax temperature parameter for contrastive objective.
            lr: Optimizing learning rate.
            weight_decay: L2 parameter regularizer weight.
            warmup_ratio: Fraction of total steps spent in linear LR warmup.
            quantizer_warmup_epochs: Epochs of continuous (quantizer-bypassed) training before
                discretization begins. k-means codebook init fires at this boundary.
            aux_ramp_epochs: Epochs over which the auxiliary-loss weights ramp 0->1 after
                quantization begins.
            loss_type: "smooth_l1" or "gaussian_nll".
            kmeans_init: Whether to run k-means initialization of codebook.
            kmeans_init_batches: Number of dataloader batches accumulated for k-means init.
            kmeans_seed: Fixed seed for reproducible k-means centroids.
            kmeans_max_samples: Optional cap on latents fed to k-means (subsampled if exceeded).
            gradient_mode: Gradient mode ("ste" or "rotation_trick").
        """
        super().__init__()
        self.save_hyperparameters()

        # Resolve the quantizer dimensionality first
        fsq_levels_resolved = None
        if quantizer_type == "fsq":
            fsq_levels_resolved = fsq_levels if fsq_levels is not None else [5, 4]
            z_dim = len(fsq_levels_resolved)
            n_states = int(np.prod(fsq_levels_resolved))

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_states = n_states
        self.quantizer_type = quantizer_type
        self.fsq_levels = fsq_levels_resolved
        self.lambda_usage = lambda_usage
        self.lambda_contrast = lambda_contrast
        self.lambda_self = lambda_self
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.quantizer_warmup_epochs = quantizer_warmup_epochs
        self.aux_ramp_epochs = aux_ramp_epochs
        self.loss_type = loss_type
        self.kmeans_init = kmeans_init
        self.kmeans_init_batches = kmeans_init_batches
        self.kmeans_seed = kmeans_seed
        self.kmeans_max_samples = kmeans_max_samples
        self.gradient_mode = gradient_mode

        # Initialize core encoder and decoder blocks
        self.encoder = ResidualMLP(input_dim, hidden_dim, z_dim, depth=3)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim, loss_type=loss_type)

        # Initialize selected quantizer backend
        if quantizer_type == "fsq":
            assert fsq_levels_resolved is not None
            self.quantizer = FSQQuantizer(fsq_levels_resolved, gradient_mode=gradient_mode)
        elif quantizer_type in ("vq", "ema_vq"):
            self.quantizer = EMAVectorQuantizer(
                n_states,
                z_dim,
                decay=decay,
                eps=eps,
                commitment_cost=commitment_cost,
                l2_normalize=l2_normalize,
                min_count=min_count,
                replacement_warmup_steps=replacement_warmup_steps,
                gradient_mode=gradient_mode,
            )
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

        # Contrastive objective projectors
        if lambda_contrast > 0.0:
            self.source_projector = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.target_projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Learnable temperature (log-scale), clamped at use to <= log(100), CLIP-style.
            self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / temperature))))

        self.loss_fn = nn.GaussianNLLLoss() if loss_type == "gaussian_nll" else None

        # Register standardization buffers for standalone scaled inference
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

        # Validation outputs collection
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def encode_states(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference pathway: maps input features to discrete state IDs.

        Args:
            x: Input features tensor of shape (N, input_dim).

        Returns:
            Tensor of indices of shape (N,).
        """
        z = self.encoder(x)
        if not isinstance(self.quantizer, EMAVectorQuantizer):
            _, indices = self.quantizer.quantize(z)
            return indices

        # fp32 lookup (shared helper) so this deterministic path matches EMA training.
        distances = _quantizer_distances(z, self.quantizer.embedding, self.quantizer.l2_normalize)
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def encode_scaled_states(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Deterministic inference pathway with standardizing scaling.

        Args:
            x_raw: Raw input features tensor of shape (N, input_dim).

        Returns:
            Tensor of indices of shape (N,).
        """
        x = (x_raw - self.feature_mean) / self.feature_std
        return self.encode_states(x)

    def init_codebook_from_data(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        n_batches: int | None = None,
    ) -> None:
        """Seed the EMA-VQ codebook from k-means of real encoder outputs.

        No-op unless the quantizer is an EMAVectorQuantizer (FSQ unaffected). Runs the
        encoder over the first ``n_batches`` batches (more than one to avoid single-batch
        bias) and fits the codebook to the concatenated latents.

        Args:
            loader: DataLoader yielding (x, y) pairs of scaled features.
            n_batches: Number of batches to accumulate (defaults to ``kmeans_init_batches``).
        """
        if not isinstance(self.quantizer, EMAVectorQuantizer):
            return

        if n_batches is None:
            n_batches = self.kmeans_init_batches

        was_training = self.training
        self.eval()
        zs: list[torch.Tensor] = []
        with torch.no_grad():
            for i, (x, _y) in enumerate(loader):
                if i >= n_batches:
                    break
                zs.append(self.encoder(x.to(self.device)))
        if was_training:
            self.train()

        if not zs:
            return
        z = torch.cat(zs, dim=0)

        # Optionally cap the number of latents fed to k-means (reproducible subsample).
        if self.kmeans_max_samples is not None and z.shape[0] > self.kmeans_max_samples:
            gen = torch.Generator(device=z.device).manual_seed(self.kmeans_seed)
            idx = torch.randperm(z.shape[0], generator=gen, device=z.device)[
                : self.kmeans_max_samples
            ]
            z = z[idx]

        self.quantizer.init_codebook(z, seed=self.kmeans_seed)

    def on_train_epoch_start(self) -> None:
        """Seed the codebook from warmed-up latents exactly at the warmup boundary.

        Fires once, when the first quantized epoch begins, so k-means runs on a useful
        latent manifold rather than an untrained encoder. No-op for FSQ.
        """
        if not self.kmeans_init:
            return
        loader = self.trainer.train_dataloader
        if (
            loader is not None
            and self.current_epoch == self.quantizer_warmup_epochs
            and isinstance(self.quantizer, EMAVectorQuantizer)
            and not bool(self.quantizer.initialized.item())
        ):
            self.init_codebook_from_data(loader)

    def _aux_ramp(self) -> float:
        """Auxiliary-loss ramp factor: 0 during warmup, rising to 1 over aux_ramp_epochs.

        Note: at the first quantized epoch the ramp is still 0, so commitment/usage/contrast
        contribute nothing that epoch while the EMA codebook already updates inside the
        quantizer (gated by ``self.training``, not by the ramp). This is intentional.
        """
        if self.current_epoch < self.quantizer_warmup_epochs:
            return 0.0
        progress = (self.current_epoch - self.quantizer_warmup_epochs) / max(
            1, self.aux_ramp_epochs
        )
        return min(1.0, max(0.0, progress))

    @property
    def _can_log(self) -> bool:
        """True when attached to a trainer with an active results collection (safe to log)."""
        return self._trainer is not None and getattr(self._trainer, "_results", None) is not None

    def forward(self, x: torch.Tensor, quantize: bool = True) -> dict[str, torch.Tensor]:
        """Standard model pass returning latent projections and losses.

        Args:
            x: Input feature tensor of shape (N, input_dim).
            quantize: If True, pass latents through the quantizer. If False (continuous
                warmup), bypass the codebook entirely: ``z_q = z``, ``vq_loss = 0``, and
                codebook stats are placeholders.

        Returns:
            Dict containing reconstruction losses, codebook metrics, and latent values.
        """
        z = self.encoder(x)
        if quantize:
            vq_loss, z_q, perplexity, indices, usage, n_replaced = self.quantizer(z)
        else:
            # Continuous warmup: decode straight from z, no codebook updates or stats.
            z_q = z
            vq_loss = torch.zeros((), device=z.device)
            indices = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
            usage = torch.zeros(self.n_states, device=z.device)
            perplexity = torch.zeros((), device=z.device)
            n_replaced = torch.zeros((), dtype=torch.long, device=z.device)
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
            "n_replaced": n_replaced,
        }

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute one training step.

        Args:
            batch: Pair of aligned inputs (x, y).
            batch_idx: Batch step offset index.

        Returns:
            Total optimized loss value.
        """
        x, y = batch

        # Continuous warmup: bypass the quantizer until quantizer_warmup_epochs.
        quantize = self.current_epoch >= self.quantizer_warmup_epochs
        aux_r = self._aux_ramp()

        out = self(x, quantize=quantize)
        vq_loss = out["vq_loss"]
        z_q_x = out["z_q"]
        perplexity = out["perplexity"]
        usage_x = out["usage"]
        mu_partner = out["mu_partner"]
        var_partner = out["var_partner"]
        mu_self = out["mu_self"]
        var_self = out["var_self"]
        n_replaced = out["n_replaced"]

        # Loss calculation
        if self.loss_fn is not None:
            # NLL term computed in fp32 for numerical stability under bf16.
            assert var_partner is not None and var_self is not None
            loss_partner = self.loss_fn(mu_partner.float(), y.float(), var_partner.float())
            loss_self = self.loss_fn(mu_self.float(), x.float(), var_self.float())
        else:
            loss_partner = F.smooth_l1_loss(mu_partner, y)
            loss_self = F.smooth_l1_loss(mu_self, x)

        recon_loss = loss_partner + self.lambda_self * loss_self

        # Entropy penalty
        usage_entropy = -(usage_x * (usage_x + 1e-10).log()).sum()
        loss_usage = -self.lambda_usage * usage_entropy

        # Contrastive objective (skip entirely while the aux ramp is zero, i.e. during warmup)
        if self.lambda_contrast > 0.0 and aux_r > 0.0:
            zq_proj = self.source_projector(z_q_x)
            h_proj = self.target_projector(y)

            zq_proj = F.normalize(zq_proj, dim=-1)
            h_proj = F.normalize(h_proj, dim=-1)

            scale = self.logit_scale.clamp(max=np.log(100.0)).exp()
            logits = scale * (zq_proj @ h_proj.t())
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_contrast = (
                self.lambda_contrast
                * 0.5
                * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            )
        else:
            loss_contrast = torch.zeros((), device=x.device)

        # Partner prediction is always full strength; auxiliary terms ramp 0->1 after warmup.
        total_loss = recon_loss + aux_r * (vq_loss + loss_usage + loss_contrast)

        # Log every objective term + per-feature-group reconstruction (catches the model
        # fitting the easy sequence dims while ignoring angular geometry).
        if self._can_log:
            with torch.no_grad():
                recon_angles = F.smooth_l1_loss(
                    mu_partner[:, 0:7].contiguous(), y[:, 0:7].contiguous()
                )
                recon_ca_distance = F.smooth_l1_loss(
                    mu_partner[:, 7:8].contiguous(), y[:, 7:8].contiguous()
                )
                recon_sequence = F.smooth_l1_loss(
                    mu_partner[:, 8:10].contiguous(), y[:, 8:10].contiguous()
                )
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("loss_total", total_loss, on_epoch=True)
            self.log("loss_partner", loss_partner, on_epoch=True)
            self.log("loss_self", loss_self, on_epoch=True)
            self.log("train_recon_loss", recon_loss, on_epoch=True)
            self.log("loss_vq", vq_loss, on_epoch=True)
            self.log("train_vq_loss", vq_loss, on_epoch=True)
            self.log("loss_usage", loss_usage, on_epoch=True)
            self.log("loss_contrast", loss_contrast, on_epoch=True)
            self.log("aux_ramp", aux_r, on_epoch=True)
            self.log("recon_angles", recon_angles, on_epoch=True)
            self.log("recon_ca_distance", recon_ca_distance, on_epoch=True)
            self.log("recon_sequence", recon_sequence, on_epoch=True)
            self.log("train_perplexity", perplexity, on_epoch=True, prog_bar=True)
            self.log("dead_codes_replaced", n_replaced.float(), prog_bar=False)

        return total_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Execute validation step metrics.

        Args:
            batch: Pair of aligned inputs (x, y).
            batch_idx: Batch step offset index.

        Returns:
            Dictionary containing metrics for the step.
        """
        x, y = batch
        out_x = self(x)
        mu_partner = out_x["mu_partner"]
        var_partner = out_x["var_partner"]
        indices_x = out_x["indices"]
        perplexity = out_x["perplexity"]
        z = out_x["z"]

        if self.loss_fn is not None:
            # NLL term computed in fp32 for numerical stability under bf16.
            assert var_partner is not None
            loss_partner = self.loss_fn(mu_partner.float(), y.float(), var_partner.float())
        else:
            loss_partner = F.smooth_l1_loss(mu_partner, y)

        # Margin calculation
        if isinstance(self.quantizer, EMAVectorQuantizer):
            distances = _quantizer_distances(
                z, self.quantizer.embedding, self.quantizer.l2_normalize
            )
            d_sorted, _ = distances.sort(dim=-1)
            margin = d_sorted[:, 1] - d_sorted[:, 0]
        else:
            z_bounded = torch.tanh(z)
            margins: list[torch.Tensor] = []
            for i, level in enumerate(self.quantizer.levels):
                scale = (level - 1) / 2 if level % 2 == 1 else level / 2
                if scale > 0:
                    dist_to_boundary = torch.abs((z_bounded[:, i] * scale) % 1.0 - 0.5) / scale
                    margins.append(dist_to_boundary)
                else:
                    margins.append(torch.ones(z.shape[0], device=z.device))
            margin = torch.stack(margins, dim=-1).min(dim=-1)[0]

        # Perturbation stability calculation (sigma=0.03)
        x_noisy = x + 0.03 * torch.randn_like(x)
        indices_noisy = self.encode_states(x_noisy)
        stability = (indices_x == indices_noisy).float().mean()

        # Target sequence discretization to evaluate mutual information
        out_y = self(y)
        indices_y = out_y["indices"]

        step_out = {
            "val_loss": loss_partner,
            "perplexity": perplexity,
            "stability": stability,
            "margin": margin.mean().detach().cpu(),
            "indices_x": indices_x.detach().cpu(),
            "indices_y": indices_y.detach().cpu(),
        }
        self.validation_step_outputs.append(step_out)
        return step_out

    def on_validation_epoch_end(self) -> None:
        """Summarize and compute validation metrics across all batches."""
        if not self.validation_step_outputs:
            return

        # Pool step metrics
        val_losses = [x["val_loss"] for x in self.validation_step_outputs]
        perplexities = [x["perplexity"] for x in self.validation_step_outputs]
        stabilities = [x["stability"] for x in self.validation_step_outputs]
        margins = [x["margin"] for x in self.validation_step_outputs]

        mean_loss = torch.stack(val_losses).mean()
        mean_perp = torch.stack(perplexities).mean()
        mean_stab = torch.stack(stabilities).mean()
        mean_margin = torch.stack(margins).mean()

        # Collect state predictions to calculate mutual information and entropy
        all_x = torch.cat([x["indices_x"] for x in self.validation_step_outputs]).numpy()
        all_y = torch.cat([x["indices_y"] for x in self.validation_step_outputs]).numpy()

        n_samples = len(all_x)
        unique_x, counts_x = np.unique(all_x, return_counts=True)

        # Calculate state usage entropy
        p_x = np.zeros(self.n_states)
        p_x[unique_x] = counts_x / n_samples
        entropy = -np.sum(p_x * np.log(p_x + 1e-10))
        normalized_entropy = entropy / np.log(self.n_states)

        dead_state_fraction = float(np.sum(p_x < 1e-5) / self.n_states)

        # Calculate mutual information of aligned pairs
        hist, _, _ = np.histogram2d(
            all_x, all_y, bins=(np.arange(self.n_states + 1), np.arange(self.n_states + 1))
        )
        p_xy = hist / hist.sum()
        p_x_marg = p_xy.sum(axis=1)
        p_y_marg = p_xy.sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            log_ratio = np.log2(p_xy / (p_x_marg[:, np.newaxis] * p_y_marg))
            aligned_mi = np.sum(p_xy * log_ratio, where=np.isfinite(log_ratio))

        # Composite validation score (lower loss, higher entropy, fewer dead states)
        val_score = -mean_loss.item() + 0.05 * normalized_entropy - 0.10 * dead_state_fraction

        # Log pooled statistics
        if self._can_log:
            self.log("val_partner_loss", mean_loss, prog_bar=True)
            self.log("val_perplexity", mean_perp, prog_bar=True)
            self.log("val_stability", mean_stab)
            self.log("val_margin", mean_margin)
            self.log("val_entropy", normalized_entropy)
            self.log("val_dead_states", dead_state_fraction)
            self.log("val_aligned_mi", aligned_mi)
            self.log("val_score", val_score, prog_bar=True)

        # Clear step cache
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Setup AdamW with no-decay bias/norm groups and warmup→cosine schedule."""
        # Decay only >=2-D matmul weights; exclude biases and LayerNorm gamma/beta.
        decay = [p for _, p in self.named_parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for _, p in self.named_parameters() if p.requires_grad and p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        # Tie schedule to the real run length, with linear warmup into cosine decay.
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(self.warmup_ratio * total_steps))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps)
                ),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def export_model(
        self,
        out_dir: Path | str,
        mean: np.ndarray,
        std: np.ndarray,
        virtual_center: tuple[float, float, float] | list[float] | None = None,
        max_ca_dist: float | None = None,
    ) -> None:
        """Export state dict, scaler configuration, and centroids to storage.

        Args:
            out_dir: Output directory path.
            mean: Feature scaler mean statistics.
            std: Feature scaler standard deviation statistics.
            virtual_center: The (alpha, beta, d) used when building the features, for
                provenance. Written as ``null`` when unknown rather than fabricated.
            max_ca_dist: The Ca-Ca distance filter used at build time, if known.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save encoder parameters
        torch.save(self.encoder.state_dict(), out_path / "encoder_state_dict.pt")

        # Save config params. Provenance fields are recorded only when known so the
        # exported config never claims a virtual center / filter that was not used.
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "z_dim": self.z_dim,
            "n_states": self.n_states,
            "quantizer_type": self.quantizer_type,
            "fsq_levels": self.fsq_levels,
            "loss_type": self.loss_type,
            "l2_normalize": getattr(self.quantizer, "l2_normalize", False),
            "gradient_mode": getattr(self.quantizer, "gradient_mode", "rotation_trick"),
            "feature_convention": "seq_delta_j_minus_i",
            "virtual_center": list(virtual_center) if virtual_center is not None else None,
            "max_ca_dist": max_ca_dist,
        }
        with open(out_path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save standardization metrics
        scaler = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        with open(out_path / "feature_scaler.json", "w") as f:
            json.dump(scaler, f, indent=2)

        # Save centroids for indexing lookups (only applicable to VQ backend)
        if self.quantizer_type in ("vq", "ema_vq") and isinstance(
            self.quantizer, EMAVectorQuantizer
        ):
            centroids = self.quantizer.embedding.detach().cpu().numpy()
            np.save(out_path / "centroids.npy", centroids)
        elif self.quantizer_type == "fsq":
            with open(out_path / "fsq_levels.json", "w") as f:
                json.dump({"levels": self.fsq_levels}, f, indent=2)

    @classmethod
    def load_from_export(
        cls, export_dir: Path | str
    ) -> tuple["TdiV2Model", np.ndarray, np.ndarray]:
        """Load a TdiV2Model from an exported directory.

        Args:
            export_dir: Path to directory containing exported model artifacts.

        Returns:
            Tuple of (loaded_model, mean_array, std_array).
        """
        export_path = Path(export_dir)
        with open(export_path / "model_config.json") as f:
            config = json.load(f)

        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            z_dim=config["z_dim"],
            n_states=config["n_states"],
            quantizer_type=config["quantizer_type"],
            fsq_levels=config.get("fsq_levels"),
            loss_type=config.get("loss_type", "smooth_l1"),
            l2_normalize=config.get("l2_normalize", True),
            gradient_mode=config.get("gradient_mode", "rotation_trick"),
        )

        # Load encoder weights
        model.encoder.load_state_dict(
            torch.load(export_path / "encoder_state_dict.pt", map_location="cpu")
        )

        # Load scaler metrics and attach/register buffers
        with open(export_path / "feature_scaler.json") as f:
            scaler = json.load(f)
        mean_arr = np.array(scaler["mean"], dtype=np.float32)
        std_arr = np.array(scaler["std"], dtype=np.float32)

        model.register_buffer("feature_mean", torch.tensor(mean_arr))
        model.register_buffer("feature_std", torch.tensor(std_arr))

        if (
            config["quantizer_type"] in ("vq", "ema_vq")
            and (export_path / "centroids.npy").exists()
            and isinstance(model.quantizer, EMAVectorQuantizer)
        ):
            centroids = np.load(export_path / "centroids.npy")
            model.quantizer.embedding.data = torch.tensor(centroids)

        model.eval()
        return model, mean_arr, std_arr
