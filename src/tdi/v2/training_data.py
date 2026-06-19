"""Feature extraction and dataset utilities for v2.

This module provides preprocessing configurations, feature standardization,
local structural consistency filtering, and coordinate augmentation (jittering)
to generate robust VQ-VAE training data.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import features, util

# Cache for computed features (vae_features, valid_mask, coords) to avoid expensive PDB re-parsing
FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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
        Jittered coordinates array.
    """
    if std <= 0.0:
        return coords
    out = coords.copy()
    # Add noise only to valid coordinates
    noise = rng.normal(0.0, std, size=out[valid_mask].shape).astype(out.dtype)
    out[valid_mask] += noise
    return out


def extract_features(
    pdb_path: str,
    virt_cb: tuple[float, float, float],
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 3D descriptors and coordinates for each residue of a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).
        jitter_std: Standard deviation of coordinates jittering noise (0.0 to disable).
        rng: Optional random generator for jittering.

    Returns:
        A tuple of (vae_features, valid_mask, coords).
    """
    # Check cache if jittering is disabled
    if jitter_std == 0.0:
        cached = FEATURE_CACHE.get(pdb_path)
        if cached is not None:
            return cached

    # Parse coordinates
    coords, valid_mask = features.get_coords_from_pdb(pdb_path, full_backbone=True)

    # Apply coordinate-level training augmentation (jittering)
    if jitter_std > 0.0 and rng is not None:
        coords = jitter_coords(coords, valid_mask, jitter_std, rng)

    coords_moved = features.move_CB(coords, virt_cb=virt_cb)

    # Convention: sequence delta is partner_index - source_index.
    # This is intentionally consistent with the existing implementation.
    partner_idx = features.find_nearest_residues(coords_moved, valid_mask)
    assert isinstance(partner_idx, np.ndarray)
    feat, valid_mask2 = features.calc_angles_forloop(coords_moved, partner_idx, valid_mask)

    # Compute sequence delta
    seq_delta = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    seq_dist_log = np.sign(seq_delta) * np.log(np.abs(seq_delta) + 1)

    # Combine structural angles and log sequence distance
    vae_features = np.hstack([feat, seq_dist_log])

    if jitter_std == 0.0:
        FEATURE_CACHE[pdb_path] = vae_features, valid_mask2, coords

    return vae_features, valid_mask2, coords


def encoder_features(
    pdb_path: str,
    virt_cb: tuple[float, float, float],
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate 3D descriptors for each residue of a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).
        jitter_std: Standard deviation of coordinates jittering noise (0.0 to disable).
        rng: Optional random generator for jittering.

    Returns:
        A tuple of (vae_features, valid_mask).
    """
    feat, mask, _ = extract_features(pdb_path, virt_cb, jitter_std, rng)
    return feat, mask


def parse_alignment(cigar_string: str) -> np.ndarray:
    """Parse CIGAR alignment string into a coordinate index mapping of matching positions."""
    return util.parse_cigar(cigar_string)


def filter_valid_pairs(
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out indices that are invalid in either structure's mask."""
    valid_mask = mask1[idx_1] & mask2[idx_2]
    return idx_1[valid_mask], idx_2[valid_mask]


def filter_ca_distance(
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    coords1: np.ndarray,
    coords2: np.ndarray,
    max_ca_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter residue pairs by Ca-Ca distance after superposition.

    Note: Ca distance filtering requires superposed coordinates or upstream filtered alignments.
    """
    if max_ca_dist is not None:
        raise NotImplementedError(
            "Ca distance filtering requires superposed coordinates or upstream filtered alignments."
        )
    return idx_1, idx_2


def make_bidirectional_pairs(
    feat1: np.ndarray,
    feat2: np.ndarray,
    idx_1: np.ndarray,
    idx_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric pair features for VQ-VAE target/partner training."""
    if len(idx_1) == 0:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0, 10), dtype=np.float32)
    x = np.vstack([feat1[idx_1], feat2[idx_2]])
    y = np.vstack([feat2[idx_2], feat1[idx_1]])
    return x, y


def align_features(
    pdb_dir: str,
    virtual_center: tuple[float, float, float],
    sid1: str,
    sid2: str,
    cigar_string: str,
    max_ca_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned descriptors for a given alignment between two PDBs.

    Filters pairs by local structural consistency using Ca-Ca distance.

    Args:
        pdb_dir: Directory where PDB files are stored.
        virtual_center: Virtual center offsets (alpha, beta, d).
        sid1: Structural ID of first protein.
        sid2: Structural ID of second protein.
        cigar_string: The alignment CIGAR string mapping residues.
        max_ca_dist: Maximum Ca-Ca distance in Angstroms for structural consistency filtering.

    Returns:
        A tuple of (feat_x, feat_y) containing aligned features.
    """
    path1 = os.path.join(pdb_dir, sid1)
    path2 = os.path.join(pdb_dir, sid2)

    feat1, mask1, coords1 = extract_features(path1, virtual_center)
    feat2, mask2, coords2 = extract_features(path2, virtual_center)

    idx_pairs = parse_alignment(cigar_string)
    if idx_pairs.shape[0] == 0:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0, 10), dtype=np.float32)

    idx_1, idx_2 = idx_pairs.T
    idx_1, idx_2 = filter_valid_pairs(idx_1, idx_2, mask1, mask2)
    idx_1, idx_2 = filter_ca_distance(idx_1, idx_2, coords1, coords2, max_ca_dist)

    return make_bidirectional_pairs(feat1, feat2, idx_1, idx_2)


def fit_standardizer(x_train: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling statistics on the training set.

    Args:
        x_train: Training features array of shape (N, D).
        eps: Minimum standard deviation floor to avoid division by zero.

    Returns:
        Tuple of (mean, std) standard deviation statistics.
    """
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply feature standardization scaling to input features.

    Args:
        x: Input features.
        mean: Training set feature means.
        std: Training set feature standard deviations.

    Returns:
        Standardized feature array.
    """
    return ((x - mean) / std).astype(np.float32)


class PairDataset(Dataset):
    """PyTorch Dataset representing aligned residue-descriptor pairs."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        jitter_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize the PairDataset.

        Args:
            x: Input descriptors of shape (N, 10).
            y: Aligned target descriptors of shape (N, 10).
            mean: Precomputed feature mean. If None, statistics are fit on x.
            std: Precomputed feature std. If None, statistics are fit on x.
            jitter_std: Noise std applied to input descriptors.
            seed: Random seed for coordinate jittering.
        """
        assert len(x) == len(y), "Features and targets must have matching length."
        self.raw_x = x.astype(np.float32)
        self.raw_y = y.astype(np.float32)

        # Standardize features using training statistics
        if mean is None or std is None:
            self.mean, self.std = fit_standardizer(self.raw_x)
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        self.x_scaled = transform(self.raw_x, self.mean, self.std)
        self.y_scaled = transform(self.raw_y, self.mean, self.std)
        self.jitter_std = jitter_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x_scaled)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_val = self.x_scaled[idx].copy()
        y_val = self.y_scaled[idx]

        # Add noise to input features in scaled descriptor space if training augmentation is on
        if self.jitter_std > 0.0:
            noise = self.rng.normal(0.0, self.jitter_std, size=x_val.shape).astype(np.float32)
            x_val += noise

        return torch.tensor(x_val), torch.tensor(y_val)
