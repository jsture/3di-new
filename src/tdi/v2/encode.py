"""Encoding utilities: convert PDB structures to discrete 3Di state sequences for v2.

This module provides predict, discretize, and PDB conversion utilities
utilizing modernized v2 models and standardization configurations.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import features, training_data
from .model import TdiV2Model

# 50 unique letters defining the structural alphabet states (excluding X/x)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz"


def predict(
    model: nn.Module,
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Pass input descriptors through the encoder model.

    Optional feature standardization statistics can be provided.

    Args:
        model: Trained encoder neural network or LightningModule.
        x: Input feature array (shape: (N, 10)).
        mean: Scaling mean array.
        std: Scaling standard deviation array.

    Returns:
        Continuous latent representations (shape: (N, Z)).
    """
    model.eval()
    if mean is not None and std is not None:
        x = training_data.transform(x, mean, std)

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Access encoder depending on model type
        if isinstance(model, TdiV2Model):
            encoder = model.encoder
        else:
            encoder = model
        return encoder.forward(x_tensor).detach().cpu().numpy()


def discretize(
    model: nn.Module,
    centroids: np.ndarray | None,
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Discretize continuous coordinates to indices of the nearest centroid or level.

    Args:
        model: Trained encoder model or LightningModule.
        centroids: Embedding centroid vectors (shape: (K, Z)) or None for FSQ.
        x: Input features (shape: (N, 10)).
        mean: Scaling mean array.
        std: Scaling standard deviation array.

    Returns:
        Indices of the closest states/centroids for each residue (shape: (N,)).
    """
    # If the model has encode_states method (TdiV2Model), use it directly
    if isinstance(model, TdiV2Model):
        if mean is not None and std is not None:
            x = training_data.transform(x, mean, std)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return model.encode_states(x_tensor).cpu().numpy()

    # Fallback to manual prediction and lookup
    z = predict(model, x, mean, std)

    # If centroids are not provided, default to L2 distance map or similar
    if centroids is None:
        raise ValueError("Centroids must be provided for non-FSQ fallback path.")

    distances = features.distance_matrix(z, centroids)
    return np.argmin(distances, axis=1)


def process_pdb(
    fn: str,
    encoder: nn.Module,
    centroids: np.ndarray | None,
    pdb_dir: str,
    virt: tuple[float, float, float],
    invalid_state: str,
    exclude_feat: int | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[str, str]:
    """Extract, discretize and convert PDB structure coordinates to sequence.

    Args:
        fn: Filename of the PDB domain.
        encoder: Loaded encoder network or TdiV2Model.
        centroids: Discretization centroid coordinates (None for FSQ).
        pdb_dir: Directory containing PDB files.
        virt: Virtual CB center parameter tuple (alpha, beta, d).
        invalid_state: Character symbol used to represent invalid residues.
        exclude_feat: One-based index of feature to exclude, or None.
        mean: Scaling mean array.
        std: Scaling standard deviation array.

    Returns:
        A tuple of (basename of PDB, sequence of 3Di states).
    """
    pdb_path = str(Path(pdb_dir) / fn)
    feat, mask = training_data.encoder_features(pdb_path, virt)

    if exclude_feat is not None:
        fmask = np.ones(feat.shape[1], dtype=bool)
        fmask[exclude_feat - 1] = False
        feat = feat[:, fmask]

    # Map descriptors to discrete states
    valid_states = discretize(encoder, centroids, feat[mask], mean, std)

    states = np.full(len(mask), -1)
    states[mask] = valid_states

    # Convert numeric state IDs to alphabet letters
    seq = "".join(LETTERS[state] if state != -1 else invalid_state for state in states)
    return Path(fn).name, seq
