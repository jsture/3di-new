"""Slim training configuration for the single-path v2 trainer.

One quantizer per run, a fixed-LR plain loop by default, and a self-describing export. The
config is small and nested only enough to keep ``--section.key`` dotted overrides ergonomic.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
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
    rotation_trick: bool = False  # VQ only; standard STE remains the default


@dataclass
class LoopConfig:
    """Plain training-loop settings (fixed LR by default)."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 5
    scheduler: str = "none"  # "none" (fixed LR) or "cosine"
    optimizer: str = "adamw"  # "adamw" or "schedulefree" (Schedule-Free AdamW)
    sf_warmup_steps: int = 500  # schedule-free: linear LR warmup steps; ignored by adamw
    sf_beta: float = 0.9  # schedule-free: momentum interpolation beta_1; ignored by adamw
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


_SECTION_TYPES = {
    "model": ModelConfig,
    "train": LoopConfig,
    "data": DataConfig,
    "outputs": OutputsConfig,
}


def _validate_train_config(cfg: TrainConfig) -> None:
    """Reject invalid settings before they can silently change training semantics."""
    model = cfg.model
    loop = cfg.train

    if model.quantizer not in {"vq", "ema_vq", "fsq"}:
        raise ValueError(
            f"model.quantizer must be 'vq', 'ema_vq', or 'fsq', got {model.quantizer!r}"
        )
    if model.loss not in {"smooth_l1", "mse"}:
        raise ValueError(f"model.loss must be 'smooth_l1' or 'mse', got {model.loss!r}")
    for name in ("input_dim", "hidden_dim", "z_dim", "n_states"):
        if getattr(model, name) <= 0:
            raise ValueError(f"model.{name} must be > 0, got {getattr(model, name)!r}")
    if not 0.0 <= model.decay < 1.0:
        raise ValueError(f"model.decay must be in [0, 1), got {model.decay!r}")
    if model.commitment_cost < 0 or model.min_count < 0:
        raise ValueError("model.commitment_cost and model.min_count must be >= 0")
    if model.replacement_warmup_steps < 0:
        raise ValueError("model.replacement_warmup_steps must be >= 0")
    if not isinstance(model.rotation_trick, bool):
        raise ValueError("model.rotation_trick must be true or false")
    if model.rotation_trick and model.quantizer == "fsq":
        raise ValueError("model.rotation_trick is only supported by the VQ quantizer")

    if loop.scheduler not in {"none", "cosine"}:
        raise ValueError(f"train.scheduler must be 'none' or 'cosine', got {loop.scheduler!r}")
    if loop.optimizer not in {"adamw", "schedulefree"}:
        raise ValueError(
            f"train.optimizer must be 'adamw' or 'schedulefree', got {loop.optimizer!r}"
        )
    # Schedule-Free exists to remove the schedule; running one on top of it is a contradiction
    # rather than a stacking of two good ideas, so reject the combination outright.
    if loop.optimizer == "schedulefree" and loop.scheduler != "none":
        raise ValueError(
            "train.optimizer='schedulefree' cannot be combined with "
            f"train.scheduler={loop.scheduler!r}; Schedule-Free replaces the LR schedule."
        )
    if loop.sf_warmup_steps < 0:
        raise ValueError(f"train.sf_warmup_steps must be >= 0, got {loop.sf_warmup_steps!r}")
    if not 0.0 <= loop.sf_beta < 1.0:
        raise ValueError(f"train.sf_beta must be in [0, 1), got {loop.sf_beta!r}")
    if loop.lr <= 0 or loop.batch_size <= 0 or loop.max_epochs <= 0:
        raise ValueError("train.lr, train.batch_size, and train.max_epochs must be > 0")
    if loop.weight_decay < 0 or loop.patience < 0:
        raise ValueError("train.weight_decay and train.patience must be >= 0")
    if loop.clip_grad_norm <= 0 or loop.kmeans_init_batches <= 0:
        raise ValueError("train.clip_grad_norm and train.kmeans_init_batches must be > 0")


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

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ValueError(f"Unknown training config section(s): {sorted(unknown_sections)}")

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            if "." not in dotted:
                raise ValueError(f"Override {dotted!r} must have the form section.key")
            section, key = dotted.split(".", 1)
            section_type = _SECTION_TYPES.get(section)
            if section_type is None:
                raise ValueError(f"Unknown training config section in override: {section!r}")
            known_keys = {item.name for item in fields(section_type)}
            if key not in known_keys:
                raise ValueError(f"Unknown training config key: {dotted!r}")
            raw.setdefault(section, {})[key] = value

    cfg = TrainConfig(
        model=ModelConfig(**raw.get("model", {})),
        train=LoopConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
        outputs=OutputsConfig(**raw.get("outputs", {})),
    )
    _validate_train_config(cfg)
    return cfg
