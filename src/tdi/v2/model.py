"""Single-path VQ-VAE model for the v2 structural alphabet.

A plain ``nn.Module`` (no PyTorch Lightning): an MLP encoder maps residue descriptors to a
latent, one of two quantizers (EMA-VQ or FSQ, selected by a flag) forms the discrete state,
and an MLP decoder predicts the aligned partner's descriptors. Training is a partner-
prediction smooth-L1 loss plus the quantizer loss, straight-through gradient, fp32 throughout.
The export is self-describing (the alphabet lives in ``config.json``) so encode/eval never
hardcode it.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .quantizers import (
    EMAVectorQuantizer,
    FSQQuantizer,
    make_quantizer,
    quantizer_distances,
)

# 50 unique letters defining the structural alphabet states (excluding the invalid state).
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz"


class ResidualMLP(nn.Module):
    """Residual Multi-Layer Perceptron with LayerNorm + SiLU blocks."""

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
        """Forward pass: (N, input_dim) -> (N, output_dim)."""
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


class AlphabetModel(nn.Module):
    """Encoder + quantizer + decoder learning a discrete structural alphabet.

    The quantizer is the alphabet-forming mechanism, selected by ``quantizer`` and never run
    as more than one per model.
    """

    quantizer: EMAVectorQuantizer | FSQQuantizer
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        z_dim: int = 4,
        n_states: int = 20,
        quantizer: str = "vq",
        levels: list[int] | None = None,
        loss: str = "smooth_l1",
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        l2_normalize: bool = True,
        min_count: float = 1.0,
        replacement_warmup_steps: int = 500,
        letters: str = LETTERS,
        invalid_state: str = "X",
    ) -> None:
        """Initialize the AlphabetModel.

        Args:
            input_dim: Dimension of input/target descriptors.
            hidden_dim: Width of the MLP hidden layers.
            z_dim: Latent dimension (overridden to ``len(levels)`` for FSQ).
            n_states: Number of discrete states (overridden to ``prod(levels)`` for FSQ).
            quantizer: ``"vq"`` (EMA vector quantization) or ``"fsq"`` (finite scalar).
            levels: Per-dimension FSQ levels (defaults to ``[5, 4]`` when FSQ).
            loss: Reconstruction loss, ``"smooth_l1"`` or ``"mse"``.
            decay: EMA decay (VQ).
            eps: Laplace smoothing epsilon (VQ).
            commitment_cost: Commitment penalty multiplier (VQ).
            l2_normalize: Cosine codebook lookup (VQ).
            min_count: Dead-code replacement threshold (VQ).
            replacement_warmup_steps: Steps before dead-code replacement begins (VQ).
            letters: The structural alphabet; recorded in the export config.
            invalid_state: Character emitted for residues with invalid coordinates.
        """
        super().__init__()

        # FSQ pins z_dim and n_states to the level grid.
        if quantizer == "fsq":
            levels = levels if levels is not None else [5, 4]
            z_dim = len(levels)
            n_states = int(np.prod(levels))

        if n_states > len(letters):
            raise ValueError(f"n_states={n_states} exceeds alphabet size {len(letters)}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_states = n_states
        self.quantizer_name = quantizer
        self.levels = levels
        self.loss = loss
        self.letters = letters
        self.invalid_state = invalid_state
        # Provenance recorded at export time; unknown until a training run sets it.
        self.virtual_center: list[float] | None = None
        self.max_ca_dist: float | None = None

        self.encoder = ResidualMLP(input_dim, hidden_dim, z_dim, depth=3)
        self.decoder = ResidualMLP(z_dim, hidden_dim, input_dim, depth=2)
        self.quantizer = make_quantizer(
            quantizer,
            n_states,
            z_dim,
            levels=levels,
            decay=decay,
            eps=eps,
            commitment_cost=commitment_cost,
            l2_normalize=l2_normalize,
            min_count=min_count,
            replacement_warmup_steps=replacement_warmup_steps,
        )

        # Standardization buffers for standalone scaled inference.
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Encode -> quantize -> decode.

        Args:
            x: Input descriptor tensor of shape (N, input_dim).

        Returns:
            Dict with ``y_hat`` (partner reconstruction), ``indices``, ``q_loss``, and the
            quantizer ``metrics`` dict.
        """
        z = self.encoder(x)
        z_q, indices, q_loss, metrics = self.quantizer(z)
        y_hat = self.decoder(z_q)
        return {
            "y_hat": y_hat,
            "indices": indices,
            "q_loss": q_loss,
            "metrics": metrics,
        }

    @torch.no_grad()
    def encode_states(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference: map input features to discrete state IDs (fp32).

        Args:
            x: Input features tensor of shape (N, input_dim).

        Returns:
            Tensor of state indices of shape (N,).
        """
        z = self.encoder(x)
        if isinstance(self.quantizer, EMAVectorQuantizer):
            distances = quantizer_distances(
                z, self.quantizer.embedding, self.quantizer.l2_normalize
            )
            return distances.argmin(dim=-1)
        _, indices = self.quantizer.quantize(z)
        return indices

    def init_codebook_from_loader(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        n_batches: int = 8,
        seed: int = 0,
    ) -> None:
        """Seed the EMA-VQ codebook from k-means of real encoder outputs.

        No-op unless the quantizer is an ``EMAVectorQuantizer`` (FSQ is unaffected). Runs the
        encoder over the first ``n_batches`` batches and fits the codebook to the latents.

        Args:
            loader: DataLoader yielding (x, y) pairs of scaled features.
            n_batches: Number of batches accumulated for k-means (more than one avoids
                single-batch bias).
            seed: Fixed seed for reproducible centroids.
        """
        if not isinstance(self.quantizer, EMAVectorQuantizer):
            return

        device = self.feature_mean.device
        was_training = self.training
        self.eval()
        zs: list[torch.Tensor] = []
        with torch.no_grad():
            for i, (x, _y) in enumerate(loader):
                if i >= n_batches:
                    break
                zs.append(self.encoder(x.to(device)))
        if was_training:
            self.train()

        if not zs:
            return
        z = torch.cat(zs, dim=0)
        self.quantizer.init_codebook(z, seed=seed)

    def save(
        self,
        out_dir: Path | str,
        mean: np.ndarray,
        std: np.ndarray,
        virtual_center: tuple[float, float, float] | list[float] | None = None,
        max_ca_dist: float | None = None,
    ) -> None:
        """Write a self-describing export: weights, config, scaler, and codebook/levels.

        Args:
            out_dir: Output directory.
            mean: Feature scaler mean.
            std: Feature scaler standard deviation.
            virtual_center: The (alpha, beta, d) used to build features, for provenance.
            max_ca_dist: The Ca-Ca distance filter used at build time, if known.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(), out_path / "encoder_state_dict.pt")
        # The decoder is small; save it unconditionally so an export supports
        # reconstruction diagnostics (what a state decodes to), not just encoding.
        torch.save(self.decoder.state_dict(), out_path / "decoder_state_dict.pt")

        # config.json is self-describing: it carries the alphabet so encode/eval never
        # hardcode it, plus build provenance recorded only when actually known.
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "z_dim": self.z_dim,
            "n_states": self.n_states,
            "quantizer": self.quantizer_name,
            "levels": self.levels,
            "loss": self.loss,
            "l2_normalize": getattr(self.quantizer, "l2_normalize", False),
            "feature_convention": "seq_delta_j_minus_i",
            "letters": self.letters,
            "invalid_state": self.invalid_state,
            "virtual_center": list(virtual_center) if virtual_center is not None else None,
            "max_ca_dist": max_ca_dist,
        }
        with open(out_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        with open(out_path / "scaler.json", "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

        if isinstance(self.quantizer, EMAVectorQuantizer):
            centroids = self.quantizer.embedding.detach().cpu().numpy()
            np.save(out_path / "centroids.npy", centroids)
        else:
            with open(out_path / "fsq_levels.json", "w") as f:
                json.dump({"levels": self.levels}, f, indent=2)

    @classmethod
    def load(cls, export_dir: Path | str) -> tuple["AlphabetModel", np.ndarray, np.ndarray]:
        """Load a model from a self-describing export directory.

        Args:
            export_dir: Path to a directory written by :meth:`save`.

        Returns:
            Tuple of ``(model, mean, std)``.
        """
        export_path = Path(export_dir)
        with open(export_path / "config.json") as f:
            config = json.load(f)

        quantizer = config["quantizer"]
        if quantizer not in ("vq", "ema_vq", "fsq"):
            raise ValueError(f"Unknown quantizer in config: {quantizer}")

        n_states = config["n_states"]
        levels = config.get("levels")
        if quantizer == "fsq":
            if not levels:
                raise ValueError("levels must be present in config for the FSQ quantizer.")
            expected = int(np.prod(levels))
            if n_states != expected:
                raise ValueError(
                    f"n_states ({n_states}) does not match prod(levels) ({levels}) = {expected}."
                )

        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            z_dim=config["z_dim"],
            n_states=n_states,
            quantizer=quantizer,
            levels=levels,
            loss=config.get("loss", "smooth_l1"),
            l2_normalize=config.get("l2_normalize", True),
            letters=config.get("letters", LETTERS),
            invalid_state=config.get("invalid_state", "X"),
        )
        model.virtual_center = config.get("virtual_center")
        model.max_ca_dist = config.get("max_ca_dist")

        encoder_path = export_path / "encoder_state_dict.pt"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder state dict not found: {encoder_path}")
        model.encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )

        decoder_path = export_path / "decoder_state_dict.pt"
        if decoder_path.exists():
            model.decoder.load_state_dict(
                torch.load(decoder_path, map_location="cpu", weights_only=True)
            )

        scaler_path = export_path / "scaler.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler configuration not found: {scaler_path}")
        with open(scaler_path) as f:
            scaler = json.load(f)
        mean_arr = np.array(scaler["mean"], dtype=np.float32)
        std_arr = np.array(scaler["std"], dtype=np.float32)

        input_dim = config["input_dim"]
        if len(mean_arr) != input_dim:
            raise ValueError(
                f"Feature mean length ({len(mean_arr)}) does not match input_dim ({input_dim})."
            )
        if len(std_arr) != input_dim:
            raise ValueError(
                f"Feature std length ({len(std_arr)}) does not match input_dim ({input_dim})."
            )
        if not np.isfinite(mean_arr).all():
            raise ValueError("Feature mean contains non-finite values.")
        if not np.isfinite(std_arr).all():
            raise ValueError("Feature std contains non-finite values.")
        if (std_arr <= 0).any():
            raise ValueError("Feature std contains non-positive values.")

        model.register_buffer("feature_mean", torch.tensor(mean_arr))
        model.register_buffer("feature_std", torch.tensor(std_arr))

        if quantizer in ("vq", "ema_vq"):
            centroids_path = export_path / "centroids.npy"
            if not centroids_path.exists():
                raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
            centroids = np.load(centroids_path)
            if len(centroids) != n_states:
                raise ValueError(
                    f"Number of centroids ({len(centroids)}) does not match n_states ({n_states})."
                )
            z_dim = config["z_dim"]
            if centroids.shape[1] != z_dim:
                raise ValueError(
                    f"Centroids dimension ({centroids.shape[1]}) does not match z_dim ({z_dim})."
                )
            if isinstance(model.quantizer, EMAVectorQuantizer):
                model.quantizer.embedding.copy_(torch.tensor(centroids))

        model.eval()
        return model, mean_arr, std_arr
