"""Declarative configuration for the tdi.data preprocessing pipeline.

Loads a YAML config into typed dataclasses, applies optional CLI overrides, and
exposes a stable content hash so identical configs produce identical outputs.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    """Input locations for one dataset."""

    name: str
    pdb_dir: str
    train_pairfile: str
    val_pairfile: str
    scop_lookup: str | None = None


@dataclass
class FeaturesConfig:
    """Feature-extraction parameters."""

    virtual_center: tuple[float, float, float] = (270.0, 0.0, 2.0)
    sequence_delta_convention: str = "j_minus_i"
    max_ca_dist: float = 5.0


@dataclass
class SamplingConfig:
    """Pair sub-sampling parameters."""

    max_pairs_per_alignment: int | None = None
    seed: int = 123


@dataclass
class OutputsConfig:
    """Output location."""

    out_dir: str


@dataclass
class PreprocessingConfig:
    """Exception/skip threshold parameters."""

    fail_on_skipped_alignments: bool = False
    max_skipped_fraction: float = 0.01


@dataclass
class DataConfig:
    """Top-level preprocessing config."""

    dataset: DatasetConfig
    outputs: OutputsConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (JSON/YAML serializable)."""
        return asdict(self)

    def config_hash(self) -> str:
        """Deterministic sha256 over the resolved config (key-sorted JSON)."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> DataConfig:
    """Load a YAML config into a DataConfig, applying optional flat overrides.

    Args:
        path: Path to the YAML config file.
        overrides: Optional ``{"section.key": value}`` map; only non-None values apply.

    Returns:
        Resolved DataConfig.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, key = dotted.split(".", 1)
            raw.setdefault(section, {})[key] = value

    features_raw = raw.get("features", {})
    vc = features_raw.get("virtual_center", [270.0, 0.0, 2.0])
    return DataConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        outputs=OutputsConfig(**raw["outputs"]),
        features=FeaturesConfig(
            virtual_center=(float(vc[0]), float(vc[1]), float(vc[2])),
            sequence_delta_convention=features_raw.get("sequence_delta_convention", "j_minus_i"),
            max_ca_dist=float(features_raw.get("max_ca_dist", 5.0)),
        ),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
    )
