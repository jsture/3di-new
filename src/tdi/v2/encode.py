"""Encoding utilities: convert PDB structures to discrete 3Di state sequences for v2.

This module provides predict, discretize, and PDB conversion utilities
utilizing modernized v2 models and standardization configurations.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import features, training_data
from .model import LETTERS, AlphabetModel


def _model_device(model: nn.Module) -> torch.device:
    """Resolve the device a model lives on, defaulting to CPU.

    Inputs are moved here before the forward pass so encoding runs on the GPU when the model
    is on one, while staying on CPU by default (the CLI must run without a GPU).
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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
    was_training = model.training
    model.eval()
    if mean is not None and std is not None:
        x = training_data.transform(x, mean, std)

    try:
        with torch.no_grad():
            x_tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(
                _model_device(model)
            )
            # Access encoder depending on model type
            if isinstance(model, AlphabetModel):
                encoder = model.encoder
            else:
                encoder = model
            return encoder.forward(x_tensor).detach().cpu().numpy()
    finally:
        # Restore prior mode so this library helper has no lasting side effect.
        if was_training:
            model.train()


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
    # If the model has encode_states method (AlphabetModel), use it directly
    if isinstance(model, AlphabetModel):
        if mean is not None and std is not None:
            x = training_data.transform(x, mean, std)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(_model_device(model))
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
    invalid_state: str | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[str, str]:
    """Extract, discretize and convert PDB structure coordinates to sequence.

    The alphabet (``letters``) and ``invalid_state`` are read from the model when it is an
    ``AlphabetModel`` (so they come from the export's self-describing config), falling back to
    the module default for a bare encoder.

    Args:
        fn: Filename of the PDB domain.
        encoder: Loaded encoder network or AlphabetModel.
        centroids: Discretization centroid coordinates (None for FSQ).
        pdb_dir: Directory containing PDB files.
        virt: Virtual CB center parameter tuple (alpha, beta, d).
        invalid_state: Override for the invalid-residue character (else taken from the model).
        mean: Scaling mean array.
        std: Scaling standard deviation array.

    Returns:
        A tuple of (basename of PDB, sequence of 3Di states).
    """
    letters: str = getattr(encoder, "letters", LETTERS)
    invalid: str = (
        invalid_state if invalid_state is not None else getattr(encoder, "invalid_state", "X")
    )

    pdb_path = str(Path(pdb_dir) / fn)
    feat, mask = training_data.encoder_features(pdb_path, virt)

    # Map descriptors to discrete states
    valid_states = discretize(encoder, centroids, feat[mask], mean, std)

    if len(valid_states) > 0 and np.max(valid_states) >= len(letters):
        raise ValueError(
            f"State index {np.max(valid_states)} exceeds available alphabet letters "
            f"(len={len(letters)})."
        )

    states = np.full(len(mask), -1)
    states[mask] = valid_states

    # Convert numeric state IDs to alphabet letters
    seq = "".join(letters[state] if state != -1 else invalid for state in states)
    return Path(fn).name, seq
