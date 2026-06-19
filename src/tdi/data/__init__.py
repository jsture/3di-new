"""Reproducible preprocessing pipeline for v2 datasets.

Public surface: config loading, the build/validate entry points, and the
CIGAR validator. Run as ``python -m tdi.data <subcommand> --config ...``.
"""

from tdi.data.cigar import CigarValidationError, validate_cigar
from tdi.data.config import DataConfig, load_config
from tdi.data.pipeline import build_features
from tdi.data.validate import validate_dataset

__all__ = [
    "CigarValidationError",
    "DataConfig",
    "build_features",
    "load_config",
    "validate_cigar",
    "validate_dataset",
]
