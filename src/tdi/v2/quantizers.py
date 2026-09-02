"""Vector and scalar quantizers for the v2 structural alphabet, behind one interface.

Two first-class quantizers form the discrete alphabet:

- :class:`EMAVectorQuantizer` -- the reference learner: EMA codebook updates, commitment
  loss, L2-normalized (cosine) lookup, mandatory dead-code replacement, k-means init.
- :class:`FSQQuantizer` -- a fixed finite-scalar-quantization comparator with no learned
  codebook.

Both return ``(z_q, indices, q_loss, metrics)`` where ``metrics`` is a dict of optional
diagnostics, so adding or dropping a metric never churns call sites. EMA-VQ uses the standard
straight-through estimator; FSQ preserves the derivative of its bounding function and applies
the straight-through estimator only to rounding, following the reference formulation. The
rotation trick has been removed from the core and lives in git history.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass while preserving the input gradient."""
    return z + (torch.round(z) - z).detach()


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
            # Intentional: accumulate raw (unnormalized) latents. The codebook is re-projected
            # to the unit sphere below (when l2_normalize), and the commitment loss anchors
            # ||z|| ~= 1, so raw magnitudes only lightly weight the centroid direction.
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
        # Needs at least two codes to have a second-nearest.
        if self.n_states >= 2:
            d_sorted, _ = distances.sort(dim=-1)
            margin = (d_sorted[:, 1] - d_sorted[:, 0]).mean()
        else:
            margin = torch.zeros((), device=z.device)

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
    level_values: torch.Tensor
    implicit_codebook: torch.Tensor

    def __init__(self, levels: list[int], eps: float = 1e-3) -> None:
        """Initialize the FSQQuantizer.

        Args:
            levels: Integer quantization steps per dimension (e.g. [5, 4] for 20 states).
            eps: Margin keeping bounded values inside the outer rounding thresholds.
        """
        super().__init__()
        if not levels or any((not isinstance(level, int)) or level < 2 for level in levels):
            raise ValueError(f"FSQ levels must be integers >= 2, got {levels!r}")
        if not 0.0 < eps < 1.0:
            raise ValueError(f"FSQ eps must be in (0, 1), got {eps}")
        self.levels = levels
        self.eps = eps
        self.n_states = int(np.prod(levels))
        self.z_dim = len(levels)

        # Reference FSQ uses the first latent dimension as the least-significant digit.
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("level_values", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("implicit_codebook", self._make_implicit_codebook())

    def _make_implicit_codebook(self) -> torch.Tensor:
        return self.indices_to_codes(torch.arange(self.n_states))

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map unbounded latents just inside each dimension's rounding range."""
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected FSQ latent width {self.z_dim}, got shape {tuple(z.shape)}")

        levels = self.level_values.to(device=z.device, dtype=z.dtype)
        half_l = (levels - 1) * (1.0 - self.eps) / 2.0
        offset = torch.where(self.level_values.to(z.device) % 2 == 1, 0.0, 0.5).to(z.dtype)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Map normalized FSQ code vectors to their integer state indices."""
        if codes.shape[-1] != self.z_dim:
            raise ValueError(
                f"Expected FSQ code width {self.z_dim}, got shape {tuple(codes.shape)}"
            )
        half_width = torch.div(self.level_values, 2, rounding_mode="floor").to(codes.device)
        digits = torch.round(codes * half_width + half_width).long()
        return (digits * self.basis.to(codes.device)).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Map integer state indices back to normalized FSQ code vectors."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.basis.device)
        if torch.any(indices < 0) or torch.any(indices >= self.n_states):
            raise ValueError(f"FSQ indices must be in [0, {self.n_states})")

        digits = (
            torch.div(indices.unsqueeze(-1), self.basis, rounding_mode="floor") % self.level_values
        )
        half_width = torch.div(self.level_values, 2, rounding_mode="floor")
        return (digits - half_width).float() / half_width

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = self.bound(z)
        half_width = torch.div(self.level_values, 2, rounding_mode="floor").to(
            device=z.device, dtype=z.dtype
        )
        # Preserve the bounding-function derivative; only rounding uses the STE.
        z_q = _round_ste(bounded) / half_width
        indices = self.codes_to_indices(z_q)
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
    replacement_warmup_steps: int = 500,
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
        replacement_warmup_steps: Steps before dead-code replacement begins (VQ).

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
            replacement_warmup_steps=replacement_warmup_steps,
        )
    raise ValueError(f"Unknown quantizer: {quantizer!r}")
