"""Tests for low-priority feature-extraction hardening in tdi.v2.features.

Covers the degenerate-neighbor sentinel in ``find_nearest_residues`` and the
numerical equivalence of the ``cdist``-based ``distance_matrix`` with the
original broadcast formulation.
"""

import numpy as np

from tdi.v2.features import (
    calc_angles_forloop,
    distance_matrix,
    find_nearest_residues,
)


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
