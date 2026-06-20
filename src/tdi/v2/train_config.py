"""Slim training configuration for the single-path v2 trainer.

One quantizer per run, a fixed-LR plain loop by default, and a self-describing export. The
config is small and nested only enough to keep ``--section.key`` dotted overrides ergonomic.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture + quantizer selection."""

    quantizer: str = "vq"  # "vq" (EMA vector quantization) or "fsq" (finite scalar)
    input_dim: int = 10
    hidden_dim: int = 64
    z_dim: int = 4
    n_states: int = 20
    levels: list[int] | None = None  # FSQ levels; defaults to [5, 4] when quantizer == "fsq"
    loss: str = "smooth_l1"  # "smooth_l1" or "mse"
    commitment_cost: float = 0.25
    decay: float = 0.99
    min_count: float = 1.0
    l2_normalize: bool = True
    replacement_warmup_steps: int = 500  # VQ: steps before dead-code replacement begins


@dataclass
class LoopConfig:
    """Plain training-loop settings (fixed LR by default)."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 5
    scheduler: str = "none"  # "none" (fixed LR) or "cosine"
    clip_grad_norm: float = 1.0
    seed: int = 1
    kmeans_init: bool = True  # VQ only; one-shot k-means codebook seeding
    kmeans_seed: int = 0
    kmeans_init_batches: int = 8


@dataclass
class DataConfig:
    """Where the preprocessed arrays live."""

    processed_dir: str = "data/processed/scop_ca5_r1"


@dataclass
class OutputsConfig:
    """Where the run directory is written."""

    out_dir: str = "outputs/models/scop_v2_default"


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: LoopConfig = field(default_factory=LoopConfig)
    data: DataConfig = field(default_factory=DataConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return config as a nested dictionary."""
        return asdict(self)

    def config_hash(self) -> str:
        """Deterministic hash over the resolved training config."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load and parse a YAML training config, applying optional dotted overrides.

    Args:
        path: Path to a YAML configuration file.
        overrides: Optional overrides of the form ``{"section.key": value}``.

    Returns:
        A populated ``TrainConfig``.
    """
    import yaml

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, key = dotted.split(".", 1)
            raw.setdefault(section, {})[key] = value

    return TrainConfig(
        model=ModelConfig(**raw.get("model", {})),
        train=LoopConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
        outputs=OutputsConfig(**raw.get("outputs", {})),
    )
