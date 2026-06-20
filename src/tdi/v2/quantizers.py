"""Vector and scalar quantizers for the v2 structural alphabet, behind one interface.

Two first-class quantizers form the discrete alphabet:

- :class:`EMAVectorQuantizer` -- the reference learner: EMA codebook updates, commitment
  loss, L2-normalized (cosine) lookup, mandatory dead-code replacement, k-means init.
- :class:`FSQQuantizer` -- a fixed finite-scalar-quantization comparator with no learned
  codebook (and so no collapse to guard against).

Both return ``(z_q, indices, q_loss, metrics)`` where ``metrics`` is a dict of optional
diagnostics, so adding or dropping a metric never churns call sites. The gradient path is the
straight-through estimator only (``z_q = z + (z_q - z).detach()``); the rotation trick has
been removed from the core and lives in git history.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


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


def quantizer_distances(
    z: torch.Tensor, codebook: torch.Tensor, l2_normalize: bool
) -> torch.Tensor:
    """Squared (cosine when ``l2_normalize``) distances between latents and codebook.

    Computed in fp32 so the deterministic inference path (``encode_states``) and the
    validation margin match the codebook math used during EMA training.

    Args:
        z: Latent tensor of shape (N, z_dim).
        codebook: Codebook tensor of shape (K, z_dim).
        l2_normalize: If True, normalize both sides (cosine distance).

    Returns:
        Distance matrix of shape (N, K), in fp32.
    """
    z32 = z.float()
    cb = codebook.float()
    if l2_normalize:
        z32 = F.normalize(z32, dim=-1)
        cb = F.normalize(cb, dim=-1)
    # Distance computation: d = x^2 + y^2 - 2xy
    return z32.pow(2).sum(dim=-1, keepdim=True) + cb.pow(2).sum(dim=-1) - 2.0 * z32 @ cb.t()


class EMAVectorQuantizer(nn.Module):
    """Vector Quantizer using Exponential Moving Average (EMA) codebook updates.

    Performs L2-normalized nearest neighbor search and applies mandatory dead-code
    replacement to avoid codebook collapse. Gradient flow is the straight-through estimator.
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
    ) -> None:
        """Initialize the EMAVectorQuantizer.

        Args:
            n_states: Number of discrete states in the codebook.
            z_dim: Dimension of continuous latent space.
            decay: Exponential decay rate for moving average statistics.
            eps: Laplace smoothing epsilon for EMA codebook counts.
            commitment_cost: Loss multiplier weighting the commitment penalty.
            l2_normalize: If True, uses cosine distance (L2 normalization) for lookups.
            min_count: Minimum EMA usage count threshold for code replacement.
            replacement_warmup_steps: Internal warmup before replacing unused centroids
                (a fixed default, not a surfaced config knob).
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

        # Initialize codebook embedding weights
        embedding = torch.randn(n_states, z_dim)
        if l2_normalize:
            embedding = F.normalize(embedding, dim=-1)
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
        if self.l2_normalize:
            self.embedding.copy_(F.normalize(self.embedding, dim=-1))
        self.ema_count.fill_(1.0)
        count = self.ema_count.clamp_min(self.eps)
        self.ema_sum.copy_(self.embedding * count.unsqueeze(1))
        self.initialized.fill_(True)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize ``z`` to the nearest codebook entry.

        Args:
            z: Input continuous latents of shape (N, z_dim).

        Returns:
            Tuple of ``(z_q, indices, q_loss, metrics)`` where ``q_loss`` is the commitment
            loss and ``metrics`` holds ``perplexity``, ``n_replaced``, and ``margin``.
        """
        distances = quantizer_distances(z, self.embedding, self.l2_normalize)
        indices = distances.argmin(dim=-1)

        encodings = F.one_hot(indices, self.n_states).float()
        z_q = encodings @ self.embedding

        n_replaced = 0
        if self.training:
            self.step_counter += 1
            counts = encodings.sum(dim=0)
            sums = encodings.t() @ z.float().detach()

            # Update moving averages
            self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
            self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)

            # Laplace smoothed count updates
            total = self.ema_count.sum()
            smoothed_count = (
                (self.ema_count + self.eps) / (total + self.n_states * self.eps) * total
            )
            self.embedding.copy_(self.ema_sum / smoothed_count.unsqueeze(1))

            # Dead-code replacement (mandatory collapse prevention, not a toggle)
            if self.step_counter > self.replacement_warmup_steps:
                dead = self.ema_count < self.min_count
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    perm = torch.randperm(z.size(0), device=z.device)
                    n_to_replace = min(n_dead, z.size(0))
                    dead_indices = torch.where(dead)[0][:n_to_replace]
                    replacements = z.float().detach()[perm[:n_to_replace]]
                    self.embedding[dead_indices] = replacements
                    self.ema_count[dead_indices] = self.min_count
                    self.ema_sum[dead_indices] = replacements * self.min_count
                    n_replaced = n_to_replace

            if self.l2_normalize:
                self.embedding.copy_(F.normalize(self.embedding, dim=-1))
                count = self.ema_count.clamp_min(self.eps)
                self.ema_sum.copy_(self.embedding * count.unsqueeze(1))

        usage = encodings.mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

        # Cheap VQ margin diagnostic: gap between the nearest and second-nearest code.
        d_sorted, _ = distances.sort(dim=-1)
        margin = (d_sorted[:, 1] - d_sorted[:, 0]).mean()

        # Commitment loss regularizes the encoder toward the (detached) codebook.
        q_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        # Straight-through estimator: forward value z_q, gradient path z.
        z_q = z + (z_q - z).detach()

        metrics = {
            "perplexity": perplexity.detach(),
            "n_replaced": torch.tensor(float(n_replaced)),
            "margin": margin.detach(),
        }
        return z_q, indices, q_loss, metrics


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer (FSQ) comparator backend for v2.

    Replaces learned vector embeddings with fixed discrete scalar steps over the continuous
    latent space, so there is no codebook to collapse and no commitment loss.
    """

    basis: torch.Tensor
    implicit_codebook: torch.Tensor

    def __init__(self, levels: list[int]) -> None:
        """Initialize the FSQQuantizer.

        Args:
            levels: Integer quantization steps per dimension (e.g. [5, 4] for 20 states).
        """
        super().__init__()
        self.levels = levels
        self.n_states = int(np.prod(levels))
        self.z_dim = len(levels)

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
                val_q = torch.round(val).clamp(0, level - 1)
                mapped = val_q / (level - 1) * 2.0 - 1.0
                idx = val_q.long()
            else:
                val = (z_bounded[:, i] + 1.0 - 1.0 / level) / 2.0 * (level - 1)
                val_q = torch.round(val).clamp(0, level - 1)
                mapped = val_q / (level - 1) * (2.0 - 2.0 / level) - 1.0 + 1.0 / level
                idx = val_q.long()

            z_q_list.append(mapped)
            indices += idx * self.basis[i]

        z_q = torch.stack(z_q_list, dim=1)
        return z_q, indices

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize ``z`` to the fixed scalar grid.

        Returns:
            Tuple of ``(z_q, indices, q_loss, metrics)`` with ``q_loss == 0`` (no codebook
            commitment) and ``metrics`` holding only ``perplexity`` (no VQ margin).
        """
        z_q, indices = self.quantize(z)
        # Straight-through estimator: forward value z_q, gradient path z.
        z_q = z + (z_q - z).detach()

        encodings = F.one_hot(indices, self.n_states).float()
        usage = encodings.mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

        q_loss = torch.zeros((), device=z.device)
        metrics = {"perplexity": perplexity.detach()}
        return z_q, indices, q_loss, metrics


def make_quantizer(
    quantizer: str,
    n_states: int,
    z_dim: int,
    levels: list[int] | None = None,
    decay: float = 0.99,
    eps: float = 1e-5,
    commitment_cost: float = 0.25,
    l2_normalize: bool = True,
    min_count: float = 1.0,
) -> EMAVectorQuantizer | FSQQuantizer:
    """Build the selected quantizer behind the shared interface.

    Args:
        quantizer: ``"vq"`` (EMA vector quantization) or ``"fsq"`` (finite scalar).
        n_states: Number of discrete states (VQ codebook size).
        z_dim: Latent dimension (VQ).
        levels: Per-dimension FSQ levels (defaults to ``[5, 4]`` for FSQ).
        decay: EMA decay (VQ).
        eps: Laplace smoothing epsilon (VQ).
        commitment_cost: Commitment penalty multiplier (VQ).
        l2_normalize: Cosine lookup (VQ).
        min_count: Dead-code replacement threshold (VQ).

    Returns:
        An ``EMAVectorQuantizer`` or ``FSQQuantizer``.
    """
    if quantizer == "fsq":
        return FSQQuantizer(levels if levels is not None else [5, 4])
    if quantizer in ("vq", "ema_vq"):
        return EMAVectorQuantizer(
            n_states,
            z_dim,
            decay=decay,
            eps=eps,
            commitment_cost=commitment_cost,
            l2_normalize=l2_normalize,
            min_count=min_count,
        )
    raise ValueError(f"Unknown quantizer: {quantizer!r}")
