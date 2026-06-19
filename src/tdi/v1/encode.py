"""Encoding utilities: convert PDB structures to discrete 3Di state sequences."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import features, training_data

# 50 unique letters defining the structural alphabet states (excluding X/x)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz"


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Pass input descriptors through the encoder model.

    Args:
        model: Trained encoder neural network.
        x: Input feature array (shape: (N, 10)).

    Returns:
        Continuous latent representations (shape: (N, 2)).
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return model(x_tensor).detach().cpu().numpy()


def discretize(model: nn.Module, centroids: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Discretize continuous coordinates to indices of the nearest centroid.

    Args:
        model: Trained encoder model.
        centroids: Embedding centroid vectors (shape: (K, 2)).
        x: Input features (shape: (N, 10)).

    Returns:
        Indices of the closest centroids for each residue (shape: (N,)).
    """
    z = predict(model, x)
    distances = features.distance_matrix(z, centroids)
    return np.argmin(distances, axis=1)


def process_pdb(
    fn: str,
    encoder: nn.Module,
    centroids: np.ndarray,
    pdb_dir: str,
    virt: tuple[float, float, float],
    invalid_state: str,
    exclude_feat: int | None,
) -> tuple[str, str]:
    """Extract, discretize and convert PDB structure coordinates to sequence.

    Args:
        fn: Filename of the PDB domain.
        encoder: Loaded encoder network.
        centroids: Discretization centroid coordinates.
        pdb_dir: Directory containing PDB files.
        virt: Virtual CB center parameter tuple (alpha, beta, d).
        invalid_state: Character symbol used to represent invalid residues.
        exclude_feat: One-based index of feature to exclude, or None.

    Returns:
        A tuple of (basename of pdb, sequence of 3Di states).
    """
    pdb_path = str(Path(pdb_dir) / fn)
    feat, mask = training_data.encoder_features(pdb_path, virt)

    if exclude_feat is not None:
        fmask = np.ones(feat.shape[1], dtype=bool)
        fmask[exclude_feat - 1] = False
        feat = feat[:, fmask]

    valid_states = discretize(encoder, centroids, feat[mask])

    states = np.full(len(mask), -1)
    states[mask] = valid_states

    seq = "".join(LETTERS[state] if state != -1 else invalid_state for state in states)
    return Path(fn).name, seq
