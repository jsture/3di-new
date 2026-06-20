"""Feature extraction and dataset utilities for v2.

This module provides preprocessing configurations, feature standardization,
local structural consistency filtering, and coordinate augmentation (jittering)
to generate robust VQ-VAE training data.
"""

import hashlib
import os
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, Sampler

from . import features, util

# Cache for computed features (vae_features, valid_mask, coords) to avoid expensive PDB re-parsing.
# Keyed on (abs path, virt_cb, feature-version, convention) so different virt_cb cannot collide.
CacheKey = tuple[str, tuple[float, float, float], str, str]
FEATURE_CACHE: dict[CacheKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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
    coordinate_jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 3D descriptors and coordinates for each residue of a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).
        coordinate_jitter_std: Std of coordinate-level jittering noise (0.0 to disable).
        rng: Optional random generator for jittering.

    Returns:
        A tuple of (vae_features, valid_mask, coords). ``coords`` are the raw parsed
        coordinates (pre CB-move), so callers never observe a mutated array.
    """
    # Cache only when jittering is disabled; key includes virt_cb + version/convention tags
    # so two runs with different virt_cb do not return stale features.
    cache_key: CacheKey = (
        os.path.abspath(pdb_path),
        (float(virt_cb[0]), float(virt_cb[1]), float(virt_cb[2])),
        "features_v2",  # feature-definition version tag
        "seq_delta_j_minus_i",  # convention tag
    )
    if coordinate_jitter_std == 0.0:
        cached = FEATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    # Parse coordinates
    coords, valid_mask = features.get_coords_from_pdb(pdb_path, full_backbone=True)

    # Apply coordinate-level training augmentation (jittering)
    if coordinate_jitter_std > 0.0 and rng is not None:
        coords = jitter_coords(coords, valid_mask, coordinate_jitter_std, rng)

    # move_CB mutates its input, so move a copy and keep raw coords intact for caching.
    coords_moved = features.move_CB(coords.copy(), virt_cb=virt_cb)

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

    # Cache the raw parsed coords (CA columns 0:3 used downstream are unaffected by move_CB).
    if coordinate_jitter_std == 0.0:
        FEATURE_CACHE[cache_key] = vae_features, valid_mask2, coords

    return vae_features, valid_mask2, coords


def encoder_features(
    pdb_path: str,
    virt_cb: tuple[float, float, float],
    coordinate_jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate 3D descriptors for each residue of a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).
        coordinate_jitter_std: Std of coordinate-level jittering noise (0.0 to disable).
        rng: Optional random generator for jittering.

    Returns:
        A tuple of (vae_features, valid_mask).
    """
    feat, mask, _ = extract_features(pdb_path, virt_cb, coordinate_jitter_std, rng)
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


def _superposed_ca_distances(ca_fixed: np.ndarray, ca_moving: np.ndarray) -> np.ndarray:
    """Return distances after rigidly superposing ca_moving onto ca_fixed."""
    fixed_center = ca_fixed.mean(axis=0)
    moving_center = ca_moving.mean(axis=0)
    fixed0 = ca_fixed - fixed_center
    moving0 = ca_moving - moving_center

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rot = Rotation.align_vectors(fixed0, moving0)[0]

    moving_aligned = rot.apply(moving0) + fixed_center
    return np.linalg.norm(ca_fixed - moving_aligned, axis=1).astype(np.float32)


def filter_ca_distance(
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    coords1: np.ndarray,
    coords2: np.ndarray,
    max_ca_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str | None]:
    """Filter residue pairs by Ca-Ca distance after superposition.

    Superposes coords2 onto coords1 using the matched pairs, then filters out pairs
    whose distance exceeds max_ca_dist. Returns the aligned indices, distances, and
    any degeneracy error classification (or None if successful).
    """
    if len(idx_1) < 3:
        return idx_1, idx_2, None, "too_few_pairs"

    # Get C-alpha coordinates (columns 0:3) for the matched pairs
    P = coords1[idx_1, 0:3]
    Q = coords2[idx_2, 0:3]

    if not np.isfinite(P).all() or not np.isfinite(Q).all():
        return idx_1[:0], idx_2[:0], np.array([], dtype=np.float32), "nonfinite_coordinates"

    # Calculate centroids
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    # Center the coordinates
    p = P - centroid_P
    q = Q - centroid_Q

    # Tolerance-based rank check: centered coords must span at least 2 dimensions
    try:
        s_p = np.linalg.svd(p, compute_uv=False)
        s_q = np.linalg.svd(q, compute_uv=False)
        if len(s_p) < 2 or s_p[1] < 1e-6 or len(s_q) < 2 or s_q[1] < 1e-6:
            return (
                idx_1[:0],
                idx_2[:0],
                np.array([], dtype=np.float32),
                "rank_deficient_coordinates",
            )
    except np.linalg.LinAlgError:
        return idx_1[:0], idx_2[:0], np.array([], dtype=np.float32), "svd_failed"

    try:
        dist = _superposed_ca_distances(P, Q)
    except (ValueError, np.linalg.LinAlgError, UserWarning):
        return idx_1[:0], idx_2[:0], np.array([], dtype=np.float32), "svd_failed"

    if max_ca_dist is not None:
        mask = dist <= max_ca_dist
        return idx_1[mask], idx_2[mask], dist[mask], None

    return idx_1, idx_2, dist, None


def assert_finite_features(x: np.ndarray, name: str) -> None:
    """Validate that features contain no NaNs or Infs."""
    if not np.isfinite(x).all():
        bad = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{name} contains {bad} non-finite values")


def make_bidirectional_pairs(
    feat1: np.ndarray,
    feat2: np.ndarray,
    idx_1: np.ndarray,
    idx_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric pair features for VQ-VAE target/partner training."""
    if len(idx_1) == 0:
        dim = feat1.shape[1] if feat1.ndim == 2 else 10
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0, dim), dtype=np.float32)
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
    max_pairs: int | None = None,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return aligned descriptors for a given alignment between two PDBs.

    Filters pairs by local structural consistency using Ca-Ca distance.

    Args:
        pdb_dir: Directory where PDB files are stored.
        virtual_center: Virtual center offsets (alpha, beta, d).
        sid1: Structural ID of first protein.
        sid2: Structural ID of second protein.
        cigar_string: The alignment CIGAR string mapping residues.
        max_ca_dist: Maximum Ca-Ca distance in Angstroms for structural consistency filtering.
        max_pairs: Optional maximum bidirectional pairs to generate.
        seed: Base random seed for sub-sampling reproducibility.

    Returns:
        A tuple of (feat_x, feat_y, meta) where meta is a dictionary containing index arrays.
    """
    path1 = util.resolve_pdb_path(pdb_dir, sid1)
    path2 = util.resolve_pdb_path(pdb_dir, sid2)

    feat1, mask1, coords1 = extract_features(str(path1), virtual_center)
    feat2, mask2, coords2 = extract_features(str(path2), virtual_center)

    idx_pairs = parse_alignment(cigar_string)
    n_pairs_before_filters = idx_pairs.shape[0]
    if n_pairs_before_filters == 0:
        return (
            np.zeros((0, 10), dtype=np.float32),
            np.zeros((0, 10), dtype=np.float32),
            {
                "n_pairs_before_filters": 0,
                "n_pairs_after_descriptor_validity": 0,
                "n_pairs_after_ca_filter": 0,
                "n_pairs_after_max_pairs": 0,
                "cap_seed": None,
            },
        )

    idx_1, idx_2 = idx_pairs.T
    idx_1, idx_2 = filter_valid_pairs(idx_1, idx_2, mask1, mask2)
    n_pairs_after_descriptor_validity = len(idx_1)

    idx_1, idx_2, dists, kabsch_error = filter_ca_distance(
        idx_1, idx_2, coords1, coords2, max_ca_dist
    )
    n_pairs_after_ca_filter = len(idx_1)

    # Sub-sample before bidirectional mapping if max_pairs is set
    cap_seed = None
    if max_pairs is not None and len(idx_1) > max_pairs // 2:
        alignment_id = f"{sid1}-{sid2}"
        hasher = hashlib.sha256(f"{alignment_id}:{seed}".encode())
        cap_seed = int(hasher.hexdigest(), 16) % (2**32)

        rng = np.random.default_rng(cap_seed)
        keep_size = max_pairs // 2
        idx = rng.choice(len(idx_1), keep_size, replace=False)
        idx_1 = idx_1[idx]
        idx_2 = idx_2[idx]
        if dists is not None:
            dists = dists[idx]

    n_pairs_after_max_pairs = len(idx_1)

    x, y = make_bidirectional_pairs(feat1, feat2, idx_1, idx_2)

    if len(x) > 0:
        assert_finite_features(x, f"x features from {sid1}-{sid2}")
        assert_finite_features(y, f"y features from {sid1}-{sid2}")

    # For bidirectional pairs, indices are also bidirectional
    # First half is sid1->sid2, second half is sid2->sid1
    if len(idx_1) > 0:
        ca1 = coords1[idx_1, 0:3]
        ca2 = coords2[idx_2, 0:3]
        raw_dists = np.linalg.norm(ca1 - ca2, axis=1)

        if dists is not None:
            superposed_dists = np.concatenate([dists, dists])
        else:
            superposed_dists = np.full(len(idx_1) * 2, np.nan)

        meta = {
            "idx_source": np.concatenate([idx_1, idx_2]),
            "idx_target": np.concatenate([idx_2, idx_1]),
            "sid_source": [sid1] * len(idx_1) + [sid2] * len(idx_2),
            "sid_target": [sid2] * len(idx_1) + [sid1] * len(idx_2),
            "ca_dist_superposed": superposed_dists,
            "ca_dist_raw": np.concatenate([raw_dists, raw_dists]),
            "n_pairs_before_filters": n_pairs_before_filters,
            "n_pairs_after_descriptor_validity": n_pairs_after_descriptor_validity,
            "n_pairs_after_ca_filter": n_pairs_after_ca_filter,
            "n_pairs_after_max_pairs": n_pairs_after_max_pairs,
            "cap_seed": cap_seed,
            "kabsch_error": kabsch_error,
        }
    else:
        meta = {
            "n_pairs_before_filters": n_pairs_before_filters,
            "n_pairs_after_descriptor_validity": n_pairs_after_descriptor_validity,
            "n_pairs_after_ca_filter": n_pairs_after_ca_filter,
            "n_pairs_after_max_pairs": n_pairs_after_max_pairs,
            "cap_seed": cap_seed,
            "kabsch_error": kabsch_error,
        }

    return x, y, meta


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
        descriptor_jitter_std: float = 0.0,
        seed: int = 42,
        fit_scaler: bool = True,
    ) -> None:
        """Initialize the PairDataset.

        Args:
            x: Input descriptors of shape (N, 10).
            y: Aligned target descriptors of shape (N, 10).
            mean: Precomputed feature mean. If None, statistics are fit on x.
            std: Precomputed feature std. If None, statistics are fit on x.
            descriptor_jitter_std: Experimental noise std applied to scaled input descriptors.
                Distinct from coordinate-level jitter (see ``extract_features``); default 0.0.
            seed: Base seed for deterministic per-item jitter.
            fit_scaler: If True, fits scaler parameters (mean/std) internally when omitted.
        """
        assert len(x) == len(y), "Features and targets must have matching length."
        self.raw_x = x.astype(np.float32)
        self.raw_y = y.astype(np.float32)

        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together")

        # Standardize features using training statistics
        if mean is None:
            if not fit_scaler:
                raise ValueError("mean/std required unless fit_scaler=True")
            self.mean, self.std = fit_standardizer(self.raw_x)
        else:
            assert std is not None
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        assert np.isfinite(self.mean).all(), "Scaler mean contains non-finite values"
        assert np.isfinite(self.std).all(), "Scaler std contains non-finite values"
        assert (self.std > 0).all(), "Scaler std contains non-positive values"

        self.x_scaled = transform(self.raw_x, self.mean, self.std)
        self.y_scaled = transform(self.raw_y, self.mean, self.std)

        assert_finite_features(self.x_scaled, "x_scaled")
        assert_finite_features(self.y_scaled, "y_scaled")

        self.descriptor_jitter_std = descriptor_jitter_std
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so per-item jitter differs across epochs but stays reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.x_scaled)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_val = self.x_scaled[idx].copy()
        y_val = self.y_scaled[idx]

        # Deterministic per-item jitter: seed from (base_seed, idx, epoch) so noise is
        # identical regardless of worker count (no shared stateful RNG to fork).
        if self.descriptor_jitter_std > 0.0:
            rng = np.random.default_rng([self.seed, idx, self.epoch])
            noise = rng.normal(0.0, self.descriptor_jitter_std, size=x_val.shape).astype(np.float32)
            x_val += noise

        return torch.tensor(x_val), torch.tensor(y_val)


class AlignmentBatchSampler(Sampler[list[int]]):
    """Batch sampler drawing several distinct alignments per batch.

    Flat random sampling fills batches with correlated residues from one structure pair,
    which weakens in-batch contrastive negatives and biases code-usage statistics. This
    sampler picks ``alignments_per_batch`` distinct alignments per batch (more if any are
    small), then draws rows within them, so each batch spans many alignments. Reproducible
    under a fixed seed; vary with epoch via ``set_epoch``.

    This is stochastic sampling, not a partition: across a single epoch some rows may be
    drawn more than once and others not at all. Each yielded batch is filled to exactly
    ``batch_size`` (re-permuting alignments if a batch would otherwise fall short).
    """

    def __init__(
        self,
        alignment_ids: Sequence[object] | np.ndarray,
        batch_size: int,
        alignments_per_batch: int,
        seed: int = 0,
    ) -> None:
        """Initialize the sampler.

        Args:
            alignment_ids: Per-row alignment identifier (length == dataset length).
            batch_size: Rows per batch.
            alignments_per_batch: Target number of distinct alignments per batch.
            seed: Base seed for reproducible draws.
        """
        if alignments_per_batch < 1:
            raise ValueError("alignments_per_batch must be >= 1")
        self.batch_size = batch_size
        self.alignments_per_batch = alignments_per_batch
        self.seed = seed
        self.epoch = 0

        groups: dict[object, list[int]] = defaultdict(list)
        for idx, aid in enumerate(alignment_ids):
            groups[aid].append(idx)
        self.groups = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        self.n = len(alignment_ids)
        self.num_batches = self.n // batch_size
        # Rows drawn per alignment so that alignments_per_batch of them fill a batch.
        self.per_alignment = max(1, batch_size // alignments_per_batch)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so batch composition varies per epoch but stays reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        n_groups = len(self.groups)
        for _ in range(self.num_batches):
            order = rng.permutation(n_groups)
            batch: list[int] = []
            pos = 0
            # Consume distinct alignments until the batch is full, re-permuting the
            # alignment order if we run out before reaching batch_size.
            while len(batch) < self.batch_size:
                if pos >= n_groups:
                    order = rng.permutation(n_groups)
                    pos = 0
                members = self.groups[order[pos]]
                pos += 1
                take = min(self.per_alignment, len(members), self.batch_size - len(batch))
                sel = rng.choice(len(members), size=take, replace=False)
                batch.extend(members[sel].tolist())
            yield batch
