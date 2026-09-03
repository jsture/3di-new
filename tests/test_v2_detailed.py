"""Geometry, standardizer, substitution-matrix, and inference tests for v2.

Covers the detailed geometric calculations, fallback nearest-neighbor behavior, feature
standardization, substitution-matrix counting, and the predict/discretize inference helpers.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

from tdi.v2.encode import discretize, predict
from tdi.v2.features import (
    approx_c_beta_position,
    calc_angles_forloop,
    distance_matrix,
    find_nearest_residues,
    get_atom_coordinates,
    move_CB,
)
from tdi.v2.submat import (
    accumulate_counts,
    calc_alphabet_mi,
    chain_contribution,
    merge_columns,
    miller_madow_corrected_mi,
    mutual_information_from_counts,
)
from tdi.v2.training_data import fit_standardizer, transform
from tdi.v2.util import mutual_information

# =====================================================================
# 1. Biopython and Geometry Tests
# =====================================================================


def create_biopython_residue(
    resname: str, atoms_list: list[tuple[str, list[float]]], hetflag: str = " "
) -> Residue:
    """Create a real Biopython Residue object for testing."""
    res = Residue((hetflag, 1, " "), resname, " ")
    for idx, (name, coord) in enumerate(atoms_list):
        atom = Atom(
            name,
            np.array(coord, dtype=np.float32),
            0.0,
            1.0,
            " ",
            f" {name} ",
            idx + 1,
            element=name[0],
        )
        res.add(atom)
    return res


def test_approx_c_beta_position_geometry() -> None:
    """Verify that approx_c_beta_position returns expected geometry parameters."""
    c_alpha = np.array([0.0, 0.0, 0.0])
    n = np.array([1.0, 0.0, 0.0])
    c_carboxyl = np.array([-1.0 / 3.0, np.sqrt(8.0) / 3.0, 0.0])

    cb = approx_c_beta_position(c_alpha, n, c_carboxyl)

    # The distance from CA to CB should equal the predefined CONSTANT (1.5336)
    dist = np.linalg.norm(cb - c_alpha)
    assert np.isclose(dist, 1.5336, atol=1e-4)
    assert not np.isnan(cb).any()


def test_get_atom_coordinates_scenarios() -> None:
    """Test get_atom_coordinates parses normal, GLY, hetatm, and invalid residues."""
    res_ala = create_biopython_residue("ALA", [("CA", [1.0, 1.0, 1.0]), ("CB", [2.0, 2.0, 2.0])])
    res_gly = create_biopython_residue(
        "GLY",
        [("CA", [0.0, 0.0, 0.0]), ("N", [1.0, 0.0, 0.0]), ("C", [0.0, 1.0, 0.0])],
    )
    res_het = create_biopython_residue("ALA", [("CA", [5.0, 5.0, 5.0])], hetflag="H_GLU")
    res_invalid = create_biopython_residue("VAL", [("CB", [3.0, 3.0, 3.0])])

    chain = [res_ala, res_gly, res_het, res_invalid]
    coords, valid_mask = get_atom_coordinates(chain, verbose=True, full_backbone=False)

    assert coords.shape == (4, 6)
    assert valid_mask[0]
    assert np.allclose(coords[0, 0:3], [1.0, 1.0, 1.0])
    assert np.allclose(coords[0, 3:6], [2.0, 2.0, 2.0])
    assert valid_mask[1]  # GLY CB approximated
    assert not valid_mask[2]  # hetatm skipped
    assert not valid_mask[3]  # missing CA


def test_distance_matrix_calculation() -> None:
    """Verify pairwise distance matrix computes correct Euclidean metrics."""
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    dist = distance_matrix(a, b)
    assert dist.shape == (2, 3)
    assert np.isclose(dist[0, 0], 0.0)
    assert np.isclose(dist[1, 0], 5.0)
    assert np.isclose(dist[1, 2], 4.0)


def test_find_nearest_residues_with_fallback() -> None:
    """Verify sequence-distance masking and fallback logic in neighbor search."""
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
    valid_mask = np.array([True, True, True, True, True, True])

    neighbors_no_limit = find_nearest_residues(coords, valid_mask, min_seq_dist=1)
    assert isinstance(neighbors_no_limit, np.ndarray)
    assert neighbors_no_limit[2] == 1

    neighbors_limit = find_nearest_residues(coords, valid_mask, min_seq_dist=2, fall_back_dist=20.0)
    assert isinstance(neighbors_limit, np.ndarray)
    assert neighbors_limit[2] == 4

    neighbors_fallback = find_nearest_residues(
        coords, valid_mask, min_seq_dist=2, fall_back_dist=10.0
    )
    assert isinstance(neighbors_fallback, np.ndarray)
    assert neighbors_fallback[2] == 1


def test_calc_angles_and_loop() -> None:
    """Test 9D feature calculation and boundary exclusions."""
    coords = np.zeros((5, 12))
    coords[:, 0:3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
        ]
    )
    coords[:, 3:6] = coords[:, 0:3] + np.array([0.0, 0.0, 1.0])
    valid_mask = np.array([True, True, True, True, True])

    partner_idx = np.array([3, 3, 3, 1, 1])
    features, new_mask = calc_angles_forloop(coords, partner_idx, valid_mask)
    assert features.shape == (5, 9)
    assert not new_mask[0]
    assert not new_mask[4]
    assert new_mask[2]

    feat_2 = features[2]
    assert not np.isnan(feat_2).any()
    assert np.isclose(feat_2[8], 1.0)


def test_move_cb_spherical_coordinates() -> None:
    """Verify CB movement and virtual center coordinate calculation."""
    coords = np.zeros((3, 12))
    coords[:, 0:3] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coords[:, 3:6] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    coords[:, 6:9] = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    coords_scaled = move_CB(coords.copy(), c_alpha_beta_distance_scale=2.0)
    assert np.allclose(coords_scaled[0, 3:6], [0.0, 0.0, 2.0])

    coords_virt = move_CB(coords.copy(), virt_cb=(0.0, 0.0, 1.5))
    ca_cb_dist = np.linalg.norm(coords_virt[:, 3:6] - coords_virt[:, 0:3], axis=1)
    assert np.allclose(ca_cb_dist, 1.5)


def test_move_cb_rotation_matches_rodrigues_formula() -> None:
    """SciPy rotations preserve the former formula on valid geometry."""
    coords = np.zeros((1, 12), dtype=np.float64)
    ca = np.array([0.2, -0.4, 0.7])
    cb = np.array([1.3, 0.1, 0.9])
    n_atm = np.array([-0.1, 0.8, 1.2])
    coords[0, 0:3] = ca
    coords[0, 3:6] = cb
    coords[0, 6:9] = n_atm
    alpha, beta = np.radians([31.0, -17.0])
    distance = 1.8

    def rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle))
        )

    vector = cb - ca
    vector = rodrigues(vector, np.cross(cb - ca, n_atm - ca), alpha)
    vector = rodrigues(vector, n_atm - ca, beta)
    expected = ca + vector * distance

    moved = move_CB(coords.copy(), virt_cb=(31.0, -17.0, distance))
    assert np.allclose(moved[0, 3:6], expected)


def test_move_cb_collinear_axis_does_not_shrink_vector() -> None:
    """An undefined first axis is an identity rotation, not a cosine scaling."""
    coords = np.zeros((1, 12), dtype=np.float64)
    coords[0, 3:6] = [1.0, 0.0, 0.0]
    coords[0, 6:9] = [2.0, 0.0, 0.0]

    moved = move_CB(coords, virt_cb=(60.0, 0.0, 2.0))

    assert np.all(np.isfinite(moved[0, 3:6]))
    assert np.allclose(moved[0, 3:6], [2.0, 0.0, 0.0])


# =====================================================================
# 2. Standardizer Tests
# =====================================================================


def test_standardizer_fit_and_transform() -> None:
    """Test feature scaling fit and transform logic with epsilon floors."""
    x = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)

    mean, std = fit_standardizer(x, eps=1e-5)
    assert np.isclose(mean[0], 1.0)
    assert np.isclose(std[0], 1e-5)
    assert np.isclose(mean[1], 4.0)
    assert np.isclose(std[1], np.std(x[:, 1]))

    x_trans = transform(x, mean, std)
    assert np.allclose(x_trans[:, 0], 0.0)
    assert np.allclose(x_trans[0, 1], -1.224744871)


# =====================================================================
# 3. Substitution Matrix & Transition Tests
# =====================================================================


def test_submat_accumulation_and_mi() -> None:
    """Verify transitions accumulation and mutual information calculation."""
    sid2seq = {"sid1": "ABC", "sid2": "BCD"}
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("sid1 sid2 3P\n")
        pairfile_path = f.name

    try:
        counts, counts_prev = accumulate_counts(pairfile_path, sid2seq, letter2idx, n_letters=4)
        assert counts[0, 1] == 1
        assert counts[1, 0] == 1
        assert counts[1, 2] == 1
        assert counts[2, 3] == 1
        assert counts.sum() == 6

        assert counts_prev[1, 1] == 1
        assert counts_prev[2, 0] == 1
        assert counts_prev[2, 2] == 1
        assert counts_prev[3, 1] == 1
        assert counts_prev.sum() == 4

        mi, mi_tot = calc_alphabet_mi(counts + 1, counts_prev + 1)
        assert mi > 0.0
        assert mi_tot is not None
    finally:
        Path(pairfile_path).unlink()


def test_mutual_information_known_distributions_and_validation() -> None:
    """Mutual information handles sparse, empty, and invalid distributions."""
    assert mutual_information(np.array([[0.5, 0.0], [0.0, 0.5]])) == pytest.approx(1.0)
    assert mutual_information(np.full((2, 2), 0.25)) == pytest.approx(0.0)
    assert mutual_information(np.zeros((2, 2))) == 0.0

    with pytest.raises(ValueError, match="negative"):
        mutual_information(np.array([[1.0, -0.5]]))
    with pytest.raises(ValueError, match="finite"):
        mutual_information(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="2D"):
        mutual_information(np.array([0.5, 0.5]))


def test_merge_columns_counts_preservation() -> None:
    """Verify merge_columns consolidates matrix elements preserving summation."""
    counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    new_counts = merge_columns(counts, i=2, j=1)
    assert new_counts.shape == (2, 2)
    assert new_counts.sum() == counts.sum()


def test_merge_columns_adjusts_retained_index_after_deletion() -> None:
    """Merging a lower state into a higher one uses the shifted retained index."""
    counts = np.arange(1, 17).reshape(4, 4)
    new_counts = merge_columns(counts, i=0, j=3)
    assert new_counts.shape == (3, 3)
    assert new_counts.sum() == counts.sum()


# =====================================================================
# 4. Inference and Predict Fallback Tests
# =====================================================================


def test_inference_fallback_discretize() -> None:
    """Test predict and discretize fallback options for a bare encoder."""
    encoder = nn.Linear(10, 4)
    x = np.random.randn(8, 10).astype(np.float32)

    z = predict(encoder, x)
    assert z.shape == (8, 4)

    centroids = np.random.randn(5, 4).astype(np.float32)
    indices = discretize(encoder, centroids, x)
    assert indices.shape == (8,)
    assert np.all(indices >= 0) and np.all(indices < 5)

    with pytest.raises(ValueError, match="Centroids must be provided"):
        _ = discretize(encoder, None, x)


def test_encode_device_move_is_noop_on_cpu() -> None:
    """Device-aware encoding produces identical output after a CPU no-op move."""
    encoder = nn.Linear(10, 4)
    x = np.random.randn(8, 10).astype(np.float32)
    centroids = np.random.randn(5, 4).astype(np.float32)

    z_before = predict(encoder, x)
    idx_before = discretize(encoder, centroids, x)

    encoder.to("cpu")  # explicit no-op; inputs are moved to the model's device internally

    z_after = predict(encoder, x)
    idx_after = discretize(encoder, centroids, x)

    assert np.array_equal(z_before, z_after)
    assert np.array_equal(idx_before, idx_after)


def test_mutual_information_from_counts_matches_normalized_joint() -> None:
    """The shared helper normalizes counts itself and agrees with the raw MI function."""
    counts = np.array([[40, 10], [10, 40]])
    assert mutual_information_from_counts(counts) == pytest.approx(
        mutual_information(counts / counts.sum())
    )
    # A perfectly coupled balanced pair carries exactly one bit.
    assert mutual_information_from_counts(np.array([[50, 0], [0, 50]])) == pytest.approx(1.0)
    # Independence carries none, and an empty matrix must not divide by zero.
    assert mutual_information_from_counts(np.full((2, 2), 25)) == pytest.approx(0.0)
    assert mutual_information_from_counts(np.zeros((2, 2), dtype=int)) == 0.0


def test_miller_madow_matches_the_formula_exactly() -> None:
    """The correction must equal (m_x + m_y - m_xy - 1) / (2 N ln 2) added to the plug-in."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(6, 7))
    n = 250

    occupied_joint = int(np.count_nonzero(counts))
    occupied_x = int(np.count_nonzero(counts.sum(axis=1)))
    occupied_y = int(np.count_nonzero(counts.sum(axis=0)))
    expected = mutual_information_from_counts(counts) + (
        occupied_x + occupied_y - occupied_joint - 1
    ) / (2 * n * np.log(2))
    assert miller_madow_corrected_mi(counts, n_observations=n) == pytest.approx(expected)


def test_miller_madow_lowers_a_sparse_many_bin_estimate() -> None:
    """When the joint occupies more bins than the marginals combined, the sign is negative."""
    # 30x30 bins from 400 independent draws: occupancy is spread, so m_xy >> m_x + m_y and the
    # correction pulls the inflated plug-in estimate back toward the truth of zero.
    rng = np.random.default_rng(1)
    counts = np.zeros((30, 30), dtype=int)
    np.add.at(counts, (rng.integers(0, 30, 400), rng.integers(0, 30, 400)), 1)

    plugin = mutual_information_from_counts(counts)
    corrected = miller_madow_corrected_mi(counts, n_observations=400)
    assert corrected < plugin
    assert abs(corrected) < abs(plugin)


def test_miller_madow_raises_a_concentrated_estimate() -> None:
    """The correction is not signed: a diagonal joint pushes it upward, even past log2(K).

    Pinned deliberately. A previous version of this test assumed the correction always lowers
    MI, which is false whenever m_xy is no larger than m_x + m_y -- for a diagonal table all
    three occupancies are K, leaving a strictly positive term.
    """
    counts = np.eye(20, dtype=int)
    plugin = mutual_information_from_counts(counts)
    corrected = miller_madow_corrected_mi(counts, n_observations=20)

    assert plugin == pytest.approx(np.log2(20))
    assert corrected > plugin
    # Smaller N means a larger correction, in whichever direction the sign points.
    assert corrected > miller_madow_corrected_mi(counts, n_observations=40)


def test_miller_madow_handles_empty_counts() -> None:
    """An empty joint must not divide by zero."""
    assert miller_madow_corrected_mi(np.zeros((3, 3), dtype=int)) == 0.0


def test_chain_contribution_inverts_the_transition_adjustment() -> None:
    """mi_prev and chain_fraction must reconstruct exactly what calc_alphabet_mi subtracted."""
    counts = np.array([[40, 10], [10, 40]])
    counts_prev = np.array([[30, 20], [20, 30]])
    mi, mi_tot = calc_alphabet_mi(counts, counts_prev)

    mi_prev, chain_fraction = chain_contribution(mi, mi_tot)
    # calc_alphabet_mi computes mi_tot = mi - (1 - 0.057) * mi_prev; invert it exactly.
    assert mi_prev == pytest.approx(mutual_information_from_counts(counts_prev))
    assert chain_fraction == pytest.approx((mi - mi_tot) / mi)
    assert mi - (1 - 0.057) * mi_prev == pytest.approx(mi_tot)


def test_chain_contribution_guards_only_the_division() -> None:
    """Zero raw MI must still recover mi_prev; only the fraction is undefined there."""
    assert chain_contribution(0.0, 0.0) == (0.0, 0.0)

    # Zero aligned MI with positive lagged MI gives a negative mi_tot, and mi_prev is still
    # exactly recoverable -- guarding the whole function would silently discard it.
    mi_prev, chain_fraction = chain_contribution(0.0, -0.5)
    assert mi_prev == pytest.approx(0.5 / (1 - 0.057))
    assert chain_fraction == 0.0
