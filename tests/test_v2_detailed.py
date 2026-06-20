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
    merge_columns,
)
from tdi.v2.training_data import fit_standardizer, transform

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


def test_merge_columns_counts_preservation() -> None:
    """Verify merge_columns consolidates matrix elements preserving summation."""
    counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    new_counts = merge_columns(counts, i=2, j=1)
    assert new_counts.shape == (2, 2)
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
