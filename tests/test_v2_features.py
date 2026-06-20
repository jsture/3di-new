"""Tests for low-priority feature-extraction hardening in tdi.v2.features.

Covers the degenerate-neighbor sentinel in ``find_nearest_residues`` and the
numerical equivalence of the ``cdist``-based ``distance_matrix`` with the
original broadcast formulation.
"""

import numpy as np

from tdi.v2.features import (
    calc_angles,
    calc_angles_forloop,
    distance_matrix,
    find_nearest_residues,
)


def _scalar_calc_angles_forloop(
    coords: np.ndarray, partner_idx: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference scalar implementation (the pre-vectorization loop) for parity checks."""
    n_res = coords.shape[0]
    out = np.full((n_res, 9), np.nan, dtype=np.float32)
    for i in range(1, n_res - 1):
        if valid_mask[i - 1] and valid_mask[i] and valid_mask[i + 1]:
            j = partner_idx[i]
            if j < 0:
                continue
            if valid_mask[j + 1] and valid_mask[j - 1]:
                out[i] = calc_angles(coords, i, j)
    return out, np.asarray(~np.isnan(out).any(axis=1))


def _naive_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Original (N, N, 3) broadcast distance, kept here as the reference."""
    return np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=-1))


def test_distance_matrix_matches_broadcast() -> None:
    """cdist-based distance_matrix equals the broadcast reference."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 3))
    b = rng.standard_normal((5, 3))

    assert np.allclose(distance_matrix(a, b), _naive_distance_matrix(a, b))

    # Self-distance case used by find_nearest_residues; diagonal must be ~0.
    d_self = distance_matrix(a, a)
    assert np.allclose(d_self, _naive_distance_matrix(a, a))
    assert np.allclose(np.diag(d_self), 0.0)


def test_find_nearest_residues_degenerate_structure() -> None:
    """A tiny (3-residue) structure has no valid partner for its interior residue.

    With only three residues, the first/last columns are forced to inf and the
    self-diagonal is masked, leaving the interior residue's distance column
    entirely inf. find_nearest_residues must return the -1 sentinel rather than
    silently pairing to residue 0.
    """
    coords = np.zeros((3, 6))
    coords[:, 3:6] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    valid_mask = np.array([True, True, True])

    partners = find_nearest_residues(coords, valid_mask)
    assert isinstance(partners, np.ndarray)
    # Interior residue has no valid partner -> sentinel, not a bogus residue 0.
    assert partners[1] == -1

    # calc_angles_forloop must skip the sentinel: no residue ends up valid.
    _, new_valid_mask = calc_angles_forloop(coords, partners, valid_mask)
    assert not new_valid_mask.any()


def test_calc_angles_forloop_matches_scalar() -> None:
    """Vectorized calc_angles_forloop equals the scalar reference on a real structure."""
    rng = np.random.default_rng(7)
    n_res = 8
    coords = np.zeros((n_res, 6), dtype=np.float32)
    # Spread CA along a jittered backbone so neighbor directions are well-defined.
    coords[:, 0:3] = np.cumsum(rng.standard_normal((n_res, 3)).astype(np.float32) + 1.0, axis=0)
    coords[:, 3:6] = coords[:, 0:3] + rng.standard_normal((n_res, 3)).astype(np.float32) * 0.5
    valid_mask = np.ones(n_res, dtype=bool)

    partner_idx = find_nearest_residues(coords, valid_mask)
    assert isinstance(partner_idx, np.ndarray)

    feats_vec, mask_vec = calc_angles_forloop(coords, partner_idx, valid_mask)
    feats_ref, mask_ref = _scalar_calc_angles_forloop(coords, partner_idx, valid_mask)

    assert np.array_equal(mask_vec, mask_ref)
    assert np.allclose(feats_vec, feats_ref, equal_nan=True, atol=1e-5)
    # The structure must actually exercise the angle math (some rows valid).
    assert mask_ref.any()


def test_calc_angles_forloop_with_sentinel_partner() -> None:
    """A -1 sentinel partner is skipped identically by vectorized and scalar paths."""
    rng = np.random.default_rng(3)
    n_res = 6
    coords = np.zeros((n_res, 6), dtype=np.float32)
    coords[:, 0:3] = np.cumsum(rng.standard_normal((n_res, 3)).astype(np.float32) + 1.0, axis=0)
    coords[:, 3:6] = coords[:, 0:3]
    valid_mask = np.ones(n_res, dtype=bool)

    partner_idx = find_nearest_residues(coords, valid_mask)
    assert isinstance(partner_idx, np.ndarray)
    partner_idx = partner_idx.copy()
    partner_idx[2] = -1  # force a sentinel partner

    feats_vec, mask_vec = calc_angles_forloop(coords, partner_idx, valid_mask)
    feats_ref, mask_ref = _scalar_calc_angles_forloop(coords, partner_idx, valid_mask)

    assert np.array_equal(mask_vec, mask_ref)
    assert np.allclose(feats_vec, feats_ref, equal_nan=True, atol=1e-5)
    assert not mask_vec[2]  # sentinel row stays invalid


def test_find_nearest_residues_sqeuclidean_parity() -> None:
    """Squared-distance search returns the same partners and true (euclidean) distances.

    Mirrors the 6-residue fixture from test_v2_detailed; partner indices are invariant to
    the monotonic sqrt, and return_dist must still report true euclidean distances.
    """
    coords = np.zeros((6, 6))
    coords[:, 3:6] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 15.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    valid_mask = np.ones(6, dtype=bool)

    # Index parity with the known-good expectations.
    near = find_nearest_residues(coords, valid_mask, min_seq_dist=1)
    far = find_nearest_residues(coords, valid_mask, min_seq_dist=2, fall_back_dist=20.0)
    assert isinstance(near, np.ndarray) and isinstance(far, np.ndarray)
    assert near[2] == 1
    assert far[2] == 4

    # return_dist distances are true euclidean distances to the chosen partner.
    partners, dists = find_nearest_residues(coords, valid_mask, return_dist=True)
    full = distance_matrix(coords[:, 3:6], coords[:, 3:6])
    for i in range(6):
        if np.isfinite(dists[i]) and partners[i] >= 0:
            assert np.isclose(dists[i], full[partners[i], i], atol=1e-6)


def test_find_nearest_residues_return_dist_sentinel() -> None:
    """return_dist path reports inf distance for sentinel (no-partner) residues."""
    coords = np.zeros((3, 6))
    coords[:, 3:6] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    valid_mask = np.array([True, True, True])

    result = find_nearest_residues(coords, valid_mask, return_dist=True)
    assert isinstance(result, tuple)
    partners, dists = result
    assert partners[1] == -1
    assert not np.isfinite(dists[1])
