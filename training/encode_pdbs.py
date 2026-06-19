"""Converts structure files (PDBs) into 3Di sequence representations using a trained model."""

import argparse
from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import create_vqvae_training_data
import extract_pdb_features

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
    distances = extract_pdb_features.distance_matrix(z, centroids)
    return np.argmin(distances, axis=1)


def process_pdb(
    fn: str,
    encoder: nn.Module,
    centroids: np.ndarray,
    pdb_dir: str,
    virt: Tuple[float, float, float],
    invalid_state: str,
    exclude_feat: argparse.Namespace,
) -> Tuple[str, str]:
    """Extract, discretize and convert PDB structure coordinates to sequence.

    Args:
        fn: Filename of the PDB domain.
        encoder: Loaded encoder network.
        centroids: Discretization centroid coordinates.
        pdb_dir: Directory containing PDB files.
        virt: Virtual CB center parameter tuple (alpha, beta, d).
        invalid_state: Character symbol used to represent invalid residues.
        exclude_feat: Index of feature to omit, if any.

    Returns:
        A tuple of (basename of pdb, sequence of 3Di states).
    """
    pdb_path = str(Path(pdb_dir) / fn)
    feat, mask = create_vqvae_training_data.encoder_features(pdb_path, virt)

    if exclude_feat is not None:
        fmask = np.ones(feat.shape[1], dtype=bool)
        fmask[exclude_feat - 1] = False
        feat = feat[:, fmask]

    valid_states = discretize(encoder, centroids, feat[mask])

    states = np.full(len(mask), -1)
    states[mask] = valid_states

    seq = "".join(
        [
            LETTERS[state] if state != -1 else invalid_state
            for state in states
        ]
    )
    return Path(fn).name, seq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB structure coordinates to discrete 3Di state sequences."
    )
    parser.add_argument("encoder", type=str, help="Path to trained *.pt encoder model.")
    parser.add_argument("centroids", type=str, help="Path to states.txt centroids file.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files.")
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        required=True,
        help="Virtual center parameters (alpha, beta, d).",
    )
    parser.add_argument(
        "--invalid-state",
        type=str,
        default="X",
        help="Symbol to represent invalid or missing coordinate states.",
    )
    parser.add_argument(
        "--exclude-feat",
        type=int,
        default=None,
        help="One-based index of feature to exclude from prediction.",
    )
    args = parser.parse_args()

    # Load encoder network and state centroid embeddings
    encoder = torch.load(args.encoder, map_location="cpu")
    encoder.eval()
    centroids = np.loadtxt(args.centroids)

    # Convert virtual parameter list to tuple
    virt_cb = (args.virt[0], args.virt[1], args.virt[2])

    for line in sys.stdin:
        fn = line.rstrip("\n")
        if not fn:
            continue
        try:
            basename, seq = process_pdb(
                fn,
                encoder,
                centroids,
                args.pdb_dir,
                virt_cb,
                args.invalid_state,
                args.exclude_feat,
            )
            print(f"{basename} {seq}")
        except Exception as e:
            print(f"Error processing {fn}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
