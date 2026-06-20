import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelTrainConfig:
    """Config parameters for TdiV2Model network architecture."""

    input_dim: int = 10
    hidden_dim: int = 64
    z_dim: int = 4
    n_states: int = 20
    quantizer_type: str = "ema_vq"
    loss_type: str = "smooth_l1"
    fsq_levels: list[int] | None = None


@dataclass
class QuantizerTrainConfig:
    """Parameters specific to vector/scalar quantizer behavior."""

    l2_normalize: bool = True
    decay: float = 0.99
    commitment_cost: float = 0.25
    min_count: float = 1.0
    replacement_warmup_steps: int = 500
    gradient_mode: str = "rotation_trick"
    kmeans_init: bool = True
    # K-means sampling configuration
    kmeans_seed: int = 0
    kmeans_init_batches: int = 16
    kmeans_init_samples: int | None = 50000


@dataclass
class LossTrainConfig:
    """Loss term coefficient and temperature settings."""

    lambda_self: float = 0.05
    lambda_usage: float = 0.001
    lambda_contrast: float = 0.02
    temperature: float = 0.1


@dataclass
class TrainingLoopConfig:
    """Standard Lightning Trainer settings."""

    batch_size: int = 512
    max_epochs: int = 20
    quantizer_warmup_epochs: int = 1
    aux_ramp_epochs: int = 1
    precision: Literal[
        "16-mixed",
        "bf16-mixed",
        "32-true",
        "64-true",
        "16-true",
        "bf16-true",
    ] = "32-true"
    seed: int = 1
    accumulate_grad_batches: int = 4


@dataclass
class OptimizerConfig:
    """Learning rate, decay, and schedules."""

    lr: float = 0.001
    weight_decay: float = 0.0001
    warmup_ratio: float = 0.03
    gradient_clip_val: float = 1.0


@dataclass
class DataSplitConfig:
    """Paths and options for features loading."""

    processed_dir: str = "data/processed/scop_ca5_v1"
    descriptor_jitter_std: float = 0.0
    sampler: str = "alignment_balanced"
    alignments_per_batch: int | None = 64
    num_workers: int = 0


@dataclass
class OutputsConfig:
    """Output directories and saving behaviors."""

    out_dir: str = "outputs/models/scop_v2_default_seed1"


@dataclass
class TrainConfig:
    """Top-level model training configuration."""

    model: ModelTrainConfig = field(default_factory=ModelTrainConfig)
    quantizer: QuantizerTrainConfig = field(default_factory=QuantizerTrainConfig)
    loss: LossTrainConfig = field(default_factory=LossTrainConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataSplitConfig = field(default_factory=DataSplitConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return config as nested dictionary."""
        return asdict(self)

    def config_hash(self) -> str:
        """Deterministic hash over resolved training config."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load and parse yaml training configuration, applying optional overrides.

    Args:
        path: Path to YAML configuration file.
        overrides: Optional nested overrides of the form {"section.key": value}.

    Returns:
        TrainConfig populated and updated config.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, key = dotted.split(".", 1)
            raw.setdefault(section, {})[key] = value

    return TrainConfig(
        model=ModelTrainConfig(**raw.get("model", {})),
        quantizer=QuantizerTrainConfig(**raw.get("quantizer", {})),
        loss=LossTrainConfig(**raw.get("loss", {})),
        training=TrainingLoopConfig(**raw.get("training", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        data=DataSplitConfig(**raw.get("data", {})),
        outputs=OutputsConfig(**raw.get("outputs", {})),
    )
