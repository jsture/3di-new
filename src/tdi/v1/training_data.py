"""Feature extraction and alignment utilities for generating VQ-VAE training data."""

import os

import numpy as np

from . import features, util

# Feature cache to avoid recomputing descriptors for the same PDB file
# maps PDB path to a tuple of (features, valid_mask)
FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def encoder_features(
    pdb_path: str, virt_cb: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate 3D descriptors for each residue of a PDB file.

    Args:
        pdb_path: Absolute or relative path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).

    Returns:
        A tuple of (vae_features, valid_mask):
            - vae_features: Descriptors shape (N, 10) including local angles and seq log distance.
            - valid_mask: Boolean mask of valid residues (shape: (N,)).
    """
    cached = FEATURE_CACHE.get(pdb_path)
    if cached is not None:
        return cached

    coords, valid_mask = features.get_coords_from_pdb(pdb_path, full_backbone=True)
    coords = features.move_CB(coords, virt_cb=virt_cb)
    partner_idx = features.find_nearest_residues(coords, valid_mask)
    feat, valid_mask2 = features.calc_angles_forloop(coords, partner_idx, valid_mask)

    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([feat, log_dist])
    FEATURE_CACHE[pdb_path] = vae_features, valid_mask2

    return vae_features, valid_mask2


def align_features(
    pdb_dir: str,
    virtual_center: tuple[float, float, float],
    sid1: str,
    sid2: str,
    cigar_string: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned descriptors for a given alignment between two PDBs.

    Args:
        pdb_dir: Directory where PDB files are stored.
        virtual_center: Virtual center offsets (alpha, beta, d).
        sid1: Structural ID of first protein.
        sid2: Structural ID of second protein.
        cigar_string: The alignment CIGAR string mapping residues.

    Returns:
        A tuple of (feat_x, feat_y):
            - feat_x: Aligned features of protein 1 (shape: (M, 10)).
            - feat_y: Aligned features of protein 2 (shape: (M, 10)).
    """
    idx_1, idx_2 = util.parse_cigar(cigar_string).T

    feat1, mask1 = encoder_features(os.path.join(pdb_dir, sid1), virtual_center)
    feat2, mask2 = encoder_features(os.path.join(pdb_dir, sid2), virtual_center)

    valid_mask = mask1[idx_1] & mask2[idx_2]
    idx_1 = idx_1[valid_mask]
    idx_2 = idx_2[valid_mask]

    x = np.vstack([feat1[idx_1], feat2[idx_2]])
    y = np.vstack([feat2[idx_2], feat1[idx_1]])
    return x, y
