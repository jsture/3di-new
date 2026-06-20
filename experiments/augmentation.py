"""Quarantined experiment: coordinate-level training augmentation (jitter).

Removed from the core data path (which no longer augments), this keeps the self-contained
``jitter_coords`` helper as a runnable snapshot. Pure-numpy; depends on nothing in ``tdi.v2``.
"""

import numpy as np


def jitter_coords(
    coords: np.ndarray,
    valid_mask: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add small Gaussian noise to valid coordinates for training augmentation.

    Args:
        coords: Coordinates array of shape (N, D).
        valid_mask: Boolean mask indicating valid residues.
        std: Standard deviation of Gaussian noise.
        rng: NumPy random generator.

    Returns:
        Jittered coordinates array (the input is not mutated).
    """
    if std <= 0.0:
        return coords
    out = coords.copy()
    # Add noise only to valid coordinates.
    noise = rng.normal(0.0, std, size=out[valid_mask].shape).astype(out.dtype)
    out[valid_mask] += noise
    return out
