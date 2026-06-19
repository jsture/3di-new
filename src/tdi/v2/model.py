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


class EMAVectorQuantizer(nn.Module):
    """Vector Quantizer using Exponential Moving Average (EMA) codebook updates.

    Performs L2-normalized nearest neighbor search and implements dead-code replacement
    to avoid codebook collapse.
    """

    embedding: torch.Tensor
    ema_count: torch.Tensor
    ema_sum: torch.Tensor
    step_counter: torch.Tensor

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
            eps: Laplace smoothing epsilon.
            commitment_cost: Loss multiplier weighting the commitment penalty.
            l2_normalize: If True, uses cosine distance (L2 normalization) for lookups.
            min_count: Minimum EMA usage count threshold for code replacement.
            replacement_warmup_steps: Warmup step count before replacing unused centroids.
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

        # Initialize codebook embedding weights uniformly
        embedding = torch.randn(n_states, z_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_states))
        self.register_buffer("ema_sum", embedding.clone())
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform vector quantization.

        Args:
            z: Input continuous latents of shape (N, z_dim).

        Returns:
            Tuple of (commit_loss, z_q, perplexity, indices, usage, n_replaced).
        """
        if self.l2_normalize:
            z_lookup = F.normalize(z, dim=-1)
            codebook = F.normalize(self.embedding, dim=-1)
        else:
            z_lookup = z
            codebook = self.embedding

        # Distance computation: d = x^2 + y^2 - 2xy
        distances = (
            z_lookup.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=-1)
            - 2.0 * z_lookup @ codebook.t()
        )
        indices = distances.argmin(dim=-1)

        # Encodings matrix
        encodings = F.one_hot(indices, self.n_states).type_as(z)
        z_q = encodings @ self.embedding

        n_replaced = 0
        if self.training:
            self.step_counter += 1
            counts = encodings.sum(dim=0)
            sums = encodings.t() @ z.detach()

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
                    replacements = z.detach()[perm[:n_to_replace]]
                    self.embedding[dead_indices] = replacements
                    self.ema_count[dead_indices] = self.min_count
                    self.ema_sum[dead_indices] = replacements * self.min_count
                    n_replaced = n_to_replace

        # Commitment loss to regularize encoder space
        commit_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        # Straight-through gradient estimator
        z_q = z + (z_q - z).detach()

        usage = encodings.float().mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())

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

    def __init__(self, levels: list[int]) -> None:
        """Initialize the FSQQuantizer.

        Args:
            levels: Integer quantization steps for each dimension (e.g. [5, 4] for 20 states).
        """
        super().__init__()
        self.levels = levels
        self.n_states = int(np.prod(levels))
        self.z_dim = len(levels)

        # Coordinate basis coefficients
        basis = []
        current = 1
        for level in reversed(levels):
            basis.append(current)
            current *= level
        self.register_buffer("basis", torch.tensor(list(reversed(basis)), dtype=torch.long))
        self.register_buffer("implicit_codebook", self._make_implicit_codebook())

    def _make_implicit_codebook(self) -> torch.Tensor:
        grids = []
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

        z_q_list = []
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
        z_q = z + (z_q - z).detach()

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
        lambda_contrast: float = 0.05,
        lambda_self: float = 0.1,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_type: str = "smooth_l1",
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
            loss_type: "smooth_l1" or "gaussian_nll".
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
        self.loss_type = loss_type

        # Initialize core encoder and decoder blocks
        self.encoder = ResidualMLP(input_dim, hidden_dim, z_dim, depth=3)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim, loss_type=loss_type)

        # Initialize selected quantizer backend
        if quantizer_type == "fsq":
            assert fsq_levels_resolved is not None
            self.quantizer = FSQQuantizer(fsq_levels_resolved)
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

        self.loss_fn = nn.GaussianNLLLoss() if loss_type == "gaussian_nll" else None

        # Register standardization buffers for standalone scaled inference
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

        # Validation outputs collection
        self.validation_step_outputs = []

    @torch.no_grad()
    def encode_states(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference pathway: maps input features to discrete state IDs.

        Args:
            x: Input features tensor of shape (N, input_dim).

        Returns:
            Tensor of indices of shape (N,).
        """
        z = self.encoder(x)
        if isinstance(self.quantizer, EMAVectorQuantizer):
            if self.quantizer.l2_normalize:
                z_lookup = F.normalize(z, dim=-1)
                codebook = F.normalize(self.quantizer.embedding, dim=-1)
            else:
                z_lookup = z
                codebook = self.quantizer.embedding
        else:
            _, indices = self.quantizer.quantize(z)
            return indices

        distances = (
            z_lookup.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=-1)
            - 2.0 * z_lookup @ codebook.t()
        )
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Standard model pass returning latent projections and losses.

        Args:
            x: Input feature tensor of shape (N, input_dim).

        Returns:
            Dict containing reconstruction losses, codebook metrics, and latent values.
        """
        z = self.encoder(x)
        vq_loss, z_q, perplexity, indices, usage, n_replaced = self.quantizer(z)
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
        out = self(x)
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
            loss_partner = self.loss_fn(mu_partner, y, var_partner)
            loss_self = self.loss_fn(mu_self, x, var_self)
        else:
            loss_partner = F.smooth_l1_loss(mu_partner, y)
            loss_self = F.smooth_l1_loss(mu_self, x)

        recon_loss = loss_partner + self.lambda_self * loss_self

        # Entropy penalty
        usage_entropy = -(usage_x * (usage_x + 1e-10).log()).sum()
        loss_usage = -self.lambda_usage * usage_entropy

        # Contrastive objective
        if self.lambda_contrast > 0.0:
            zq_proj = self.source_projector(z_q_x)
            h_proj = self.target_projector(y)

            zq_proj = F.normalize(zq_proj, dim=-1)
            h_proj = F.normalize(h_proj, dim=-1)

            logits = zq_proj @ h_proj.t() / self.temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_contrast = self.lambda_contrast * F.cross_entropy(logits, labels)
        else:
            loss_contrast = 0.0

        total_loss = recon_loss + vq_loss + loss_usage + loss_contrast

        # Log training statistics
        if self._trainer is not None and getattr(self._trainer, "_results", None) is not None:
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_recon_loss", recon_loss, on_epoch=True)
            self.log("train_vq_loss", vq_loss, on_epoch=True)
            self.log("train_perplexity", perplexity, on_epoch=True, prog_bar=True)
            self.log("dead_codes_replaced", n_replaced.float(), prog_bar=False)

        return total_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
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

        if self.loss_fn is not None:
            loss_partner = self.loss_fn(mu_partner, y, var_partner)
        else:
            loss_partner = F.smooth_l1_loss(mu_partner, y)

        # Perturbation stability calculation (sigma=0.03)
        x_noisy = x + 0.03 * torch.randn_like(x)
        indices_noisy = self.encode_states(x_noisy)
        stability = (indices_x == indices_noisy).float().mean()

        # Target sequence discretization to evaluate mutual information
        out_y = self(y)
        indices_y = out_y["indices"]

        # Store outputs for epoch level validation pooling
        step_out = {
            "val_loss": loss_partner,
            "perplexity": perplexity,
            "stability": stability,
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

        mean_loss = torch.stack(val_losses).mean()
        mean_perp = torch.stack(perplexities).mean()
        mean_stab = torch.stack(stabilities).mean()

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
        if self._trainer is not None and getattr(self._trainer, "_results", None) is not None:
            self.log("val_partner_loss", mean_loss, prog_bar=True)
            self.log("val_perplexity", mean_perp, prog_bar=True)
            self.log("val_stability", mean_stab)
            self.log("val_entropy", normalized_entropy)
            self.log("val_dead_states", dead_state_fraction)
            self.log("val_aligned_mi", aligned_mi)
            self.log("val_score", val_score, prog_bar=True)

        # Clear step cache
        self.validation_step_outputs.clear()

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.CosineAnnealingLR]]:
        """Setup optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # CosineAnnealingLR for smooth learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # T_max can be set dynamically during training if needed
        )
        return [optimizer], [scheduler]

    def export_model(self, out_dir: Path | str, mean: np.ndarray, std: np.ndarray) -> None:
        """Export state dict, scaler configuration, and centroids to storage.

        Args:
            out_dir: Output directory path.
            mean: Feature scaler mean statistics.
            std: Feature scaler standard deviation statistics.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save encoder parameters
        torch.save(self.encoder.state_dict(), out_path / "encoder_state_dict.pt")

        # Save config params
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "z_dim": self.z_dim,
            "n_states": self.n_states,
            "quantizer_type": self.quantizer_type,
            "fsq_levels": self.fsq_levels,
            "loss_type": self.loss_type,
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

        model.register_buffer("feature_mean", torch.from_numpy(mean_arr))
        model.register_buffer("feature_std", torch.from_numpy(std_arr))

        if (
            config["quantizer_type"] in ("vq", "ema_vq")
            and (export_path / "centroids.npy").exists()
        ):
            centroids = np.load(export_path / "centroids.npy")
            model.quantizer.embedding.data = torch.from_numpy(centroids)

        model.eval()
        return model, mean_arr, std_arr


def create_vqvae(
    seed: int,
    input_dim: int,
    hidden_dim: int,
    z_dim: int,
    n_states: int,
    quantizer_type: str = "vq",
    loss_type: str = "smooth_l1",
) -> TdiV2Model:
    """Instantiate and initialize TdiV2Model with a fixed random seed.

    Args:
        seed: Random seed.
        input_dim: Feature width.
        hidden_dim: Projection MLP width.
        z_dim: Continuous latent space width.
        n_states: Discrete states count.
        quantizer_type: "vq" or "fsq".
        loss_type: "smooth_l1" or "gaussian_nll".

    Returns:
        Configured TdiV2Model.
    """
    torch.manual_seed(seed)
    L.seed_everything(seed)
    return TdiV2Model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        n_states=n_states,
        quantizer_type=quantizer_type,
        loss_type=loss_type,
    )
