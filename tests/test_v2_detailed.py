"""Comprehensive unit and integration tests for v2 structural VAE components.

This test file covers detailed geometric calculations, fallback nearest neighbor
behavior, custom quantizers, custom loss functions, dead-code replacement, and
substitution matrix counting.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

from tdi.v2.encode import (
    discretize,
    predict,
)
from tdi.v2.features import (
    approx_c_beta_position,
    calc_angles_forloop,
    distance_matrix,
    find_nearest_residues,
    get_atom_coordinates,
    move_CB,
)
from tdi.v2.model import (
    EMAVectorQuantizer,
    FSQQuantizer,
    TdiV2Model,
)
from tdi.v2.submat import (
    accumulate_counts,
    calc_alphabet_mi,
    merge_columns,
)
from tdi.v2.training_data import (
    fit_standardizer,
    jitter_coords,
    transform,
)

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
    # 1. Normal alanine residue
    res_ala = create_biopython_residue("ALA", [("CA", [1.0, 1.0, 1.0]), ("CB", [2.0, 2.0, 2.0])])

    # 2. Glycine residue (should trigger CB approximation if N and C are present)
    res_gly = create_biopython_residue(
        "GLY",
        [
            ("CA", [0.0, 0.0, 0.0]),
            ("N", [1.0, 0.0, 0.0]),
            ("C", [0.0, 1.0, 0.0]),
        ],
    )

    # 3. Hetatm residue (should be skipped)
    res_het = create_biopython_residue("ALA", [("CA", [5.0, 5.0, 5.0])], hetflag="H_GLU")

    # 4. Residue with missing CA (invalid)
    res_invalid = create_biopython_residue("VAL", [("CB", [3.0, 3.0, 3.0])])

    chain = [res_ala, res_gly, res_het, res_invalid]

    coords, valid_mask = get_atom_coordinates(chain, verbose=True, full_backbone=False)

    # Shape should be (4, 6)
    assert coords.shape == (4, 6)
    # Alanine should be valid
    assert valid_mask[0]
    assert np.allclose(coords[0, 0:3], [1.0, 1.0, 1.0])
    assert np.allclose(coords[0, 3:6], [2.0, 2.0, 2.0])

    # Glycine should approximate CB and be valid
    assert valid_mask[1]
    # Hetatm should be invalid (skipped)
    assert not valid_mask[2]
    # Missing CA should be invalid
    assert not valid_mask[3]


def test_distance_matrix_calculation() -> None:
    """Verify pairwise distance matrix computes correct Euclidean metrics."""
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    dist = distance_matrix(a, b)
    assert dist.shape == (2, 3)
    assert np.isclose(dist[0, 0], 0.0)
    assert np.isclose(dist[1, 0], 5.0)  # sqrt(3^2 + 4^2) = 5
    assert np.isclose(dist[1, 2], 4.0)  # distance from (3,4) to (3,0) is 4


def test_find_nearest_residues_with_fallback() -> None:
    """Verify sequence-distance masking and fallback logic in neighbor search."""
    # Let's create a coordinate array for 6 residues
    coords = np.zeros((6, 6))
    # Fill in C-beta coordinates (cols 3:6)
    coords[:, 3:6] = np.array(
        [
            [0.0, 0.0, 0.0],  # 0 (boundary)
            [0.0, 0.0, 0.0],  # 1 (valid target)
            [0.0, 1.2, 0.0],  # 2 (query)
            [0.0, 2.5, 0.0],  # 3 (sequence neighbor)
            [0.0, 15.0, 0.0],  # 4 (sequence-distant neighbor)
            [0.0, 0.0, 0.0],  # 5 (boundary)
        ]
    )
    valid_mask = np.array([True, True, True, True, True, True])

    # Without seq-distance limit (min_seq_dist=1), nearest to 2 is 1 (dist=1.2) vs 3 (dist=1.3)
    neighbors_no_limit = find_nearest_residues(coords, valid_mask, min_seq_dist=1)
    assert isinstance(neighbors_no_limit, np.ndarray)
    assert neighbors_no_limit[2] == 1

    # With min_seq_dist=2, index 1 (|1-2|=1) and 3 (|3-2|=1) are sequence neighbors.
    # The only eligible sequence-distant valid residue is index 4.
    # Case A: If fallback threshold is large (e.g. fall_back_dist=20.0), distance to 4 (13.8)
    # is less than 20.0, so it should return index 4.
    neighbors_limit = find_nearest_residues(coords, valid_mask, min_seq_dist=2, fall_back_dist=20.0)
    assert isinstance(neighbors_limit, np.ndarray)
    assert neighbors_limit[2] == 4

    # Case B: If fallback threshold is small (e.g. fall_back_dist=10.0), distance to 4 (13.8)
    # exceeds 10.0, so it should fall back to the unconstrained nearest neighbor, which is 1.
    neighbors_fallback = find_nearest_residues(
        coords, valid_mask, min_seq_dist=2, fall_back_dist=10.0
    )
    assert isinstance(neighbors_fallback, np.ndarray)
    assert neighbors_fallback[2] == 1


def test_calc_angles_and_loop() -> None:
    """Test 9D feature calculation and boundary exclusions."""
    # Create mock coordinate set for 5 residues
    coords = np.zeros((5, 12))
    # C-alphas
    coords[:, 0:3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
        ]
    )
    # C-betas
    coords[:, 3:6] = coords[:, 0:3] + np.array([0.0, 0.0, 1.0])
    valid_mask = np.array([True, True, True, True, True])

    partner_idx = np.array([3, 3, 3, 1, 1])

    features, new_mask = calc_angles_forloop(coords, partner_idx, valid_mask)
    assert features.shape == (5, 9)

    # Boundary residues (index 0 and 4) must be invalid
    # (since they lack sequence neighbors i-1 or i+1)
    assert not new_mask[0]
    assert not new_mask[4]
    assert new_mask[2]

    # Check properties of calculated features at index 2
    feat_2 = features[2]
    assert not np.isnan(feat_2).any()
    # Sequence distance clipped check: partner of 2 is 3. 3-2 = 1.
    assert np.isclose(feat_2[8], 1.0)


def test_move_cb_spherical_coordinates() -> None:
    """Verify CB movement and virtual center coordinate calculation."""
    coords = np.zeros((3, 12))
    # CA
    coords[:, 0:3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    # CB
    coords[:, 3:6] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    # N
    coords[:, 6:9] = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    # 1. Scale distance
    coords_scaled = move_CB(coords.copy(), c_alpha_beta_distance_scale=2.0)
    # CB vector was [0, 0, 1] relative to CA. Scaled should be [0, 0, 2]
    assert np.allclose(coords_scaled[0, 3:6], [0.0, 0.0, 2.0])

    # 2. Virtual spherical parameter movement (alpha=0, beta=0, d=1.5)
    coords_virt = move_CB(coords.copy(), virt_cb=(0.0, 0.0, 1.5))
    # Distance from CA to new CB should be exactly 1.5
    ca_cb_dist = np.linalg.norm(coords_virt[:, 3:6] - coords_virt[:, 0:3], axis=1)
    assert np.allclose(ca_cb_dist, 1.5)


# =====================================================================
# 2. Training Data and Standardizer Tests
# =====================================================================


def test_jitter_coords_reproducibility() -> None:
    """Verify coordinate jittering noise and mask bounds."""
    coords = np.zeros((10, 3), dtype=np.float32)
    valid_mask = np.array([True, True, True, False, False, True, True, True, True, True])

    rng = np.random.default_rng(123)
    out1 = jitter_coords(coords, valid_mask, std=0.5, rng=rng)

    # Unmasked indices should remain strictly 0
    assert np.all(out1[3:5] == 0.0)
    # Masked indices should be modified
    assert not np.all(out1[valid_mask] == 0.0)

    # Reproducibility
    rng2 = np.random.default_rng(123)
    out2 = jitter_coords(coords, valid_mask, std=0.5, rng=rng2)
    assert np.array_equal(out1, out2)


def test_standardizer_fit_and_transform() -> None:
    """Test feature scaling fit and transform logic with epsilon floors."""
    x = np.array(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
        ],
        dtype=np.float32,
    )

    mean, std = fit_standardizer(x, eps=1e-5)

    # First feature variance is 0, so std should be floored to eps
    assert np.isclose(mean[0], 1.0)
    assert np.isclose(std[0], 1e-5)

    # Second feature mean is 4, std is sqrt(((2-4)^2 + 0 + (6-4)^2)/3) = sqrt(8/3) ≈ 1.633
    assert np.isclose(mean[1], 4.0)
    assert np.isclose(std[1], np.std(x[:, 1]))

    # Transform
    x_trans = transform(x, mean, std)
    assert np.allclose(x_trans[:, 0], 0.0)
    assert np.allclose(x_trans[0, 1], -1.224744871)  # (2 - 4) / 1.633


# =====================================================================
# 3. Model Architecture and Quantizer Tests
# =====================================================================


def test_vq_dead_code_replacement() -> None:
    """Test that dead VQ centroids are successfully replaced with batch latents."""
    # 3 states, latent z_dim=2, decay=0.5, warmup=2 steps
    quantizer = EMAVectorQuantizer(
        n_states=3,
        z_dim=2,
        decay=0.5,
        commitment_cost=0.1,
        min_count=1.0,
        replacement_warmup_steps=2,
        l2_normalize=True,
    )
    quantizer.train()

    # Manually configure embeddings so that they are separated
    quantizer.embedding.copy_(
        torch.tensor(
            [
                [1.0, 1.0],  # state 0
                [-1.0, -1.0],  # state 1
                [10.0, 10.0],  # state 2 (too far, will become dead)
            ]
        )
    )
    quantizer.ema_sum.copy_(quantizer.embedding)
    quantizer.ema_count.copy_(torch.tensor([1.0, 1.0, 1.0]))

    # Target input: all vectors are close to state 0 or state 1. None close to state 2.
    # Pass 1: step = 1
    z1 = torch.tensor([[0.9, 0.9], [-0.9, -0.9]], dtype=torch.float32)
    _, _, _, _, _, n_replaced = quantizer(z1)
    assert n_replaced.item() == 0

    # Pass 2: step = 2
    _, _, _, _, _, n_replaced = quantizer(z1)
    assert n_replaced.item() == 0

    # Pass 3: step = 3 (> warmup=2). State 2 count is 0, so it is dead and should be replaced.
    # Provide a batch with a distinct vector
    z2 = torch.tensor([[5.0, -5.0], [5.0, -5.0]], dtype=torch.float32)
    _, _, _, _, _, n_replaced = quantizer(z2)

    # State 2 should be replaced by [5.0, -5.0] from batch
    assert n_replaced.item() > 0
    assert torch.allclose(quantizer.embedding[2], torch.tensor([5.0, -5.0]))


def test_fsq_quantizer_grids() -> None:
    """Test FSQ grid creation for odd vs even levels."""
    # Odd levels: [3] should produce [-1.0, 0.0, 1.0]
    q_odd = FSQQuantizer(levels=[3])
    expected_odd = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    assert torch.allclose(q_odd.implicit_codebook, expected_odd)

    # Even levels: [4] should produce grid spacing: linspace(-0.75, 0.75, 4)
    # grid: -1 + 0.25 = -0.75, 1 - 0.25 = 0.75.
    q_even = FSQQuantizer(levels=[4])
    expected_even = torch.tensor([[-0.75], [-0.25], [0.25], [0.75]], dtype=torch.float32)
    assert torch.allclose(q_even.implicit_codebook, expected_even)


def test_model_gaussian_nll_reconstruction() -> None:
    """Test Decoder and TdiV2Model output shapes and loss with Gaussian NLL."""
    model = TdiV2Model(
        input_dim=5,
        hidden_dim=32,
        z_dim=3,
        n_states=10,
        quantizer_type="vq",
        loss_type="gaussian_nll",
        lambda_self=0.2,
    )

    # Verify self-reconstruction forward pass
    x = torch.randn(8, 5)
    out = model(x)

    # When loss_type is gaussian_nll, decoder should return variance
    assert out["var_partner"] is not None
    assert out["var_self"] is not None
    assert out["var_partner"].shape == (8, 5)
    # Variance should be strictly positive (softplus output)
    assert torch.all(out["var_partner"] >= 1e-4)

    # Run a training step and verify it compiles without shape error
    y = torch.randn(8, 5)
    loss = model.training_step((x, y), 0)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_model_contrastive_loss() -> None:
    """Verify model handles contrastive loss configurations."""
    # 1. With contrastive loss enabled
    model_w_contrast = TdiV2Model(
        input_dim=5,
        hidden_dim=32,
        lambda_contrast=0.5,
    )
    assert hasattr(model_w_contrast, "source_projector")
    x = torch.randn(8, 5)
    y = torch.randn(8, 5)
    loss_val = model_w_contrast.training_step((x, y), 0)
    assert not torch.isnan(loss_val)

    # 2. Without contrastive loss enabled
    model_no_contrast = TdiV2Model(
        input_dim=5,
        hidden_dim=32,
        lambda_contrast=0.0,
    )
    assert not hasattr(model_no_contrast, "source_projector")


# =====================================================================
# 4. Substitution Matrix & Transition Tests
# =====================================================================


def test_submat_accumulation_and_mi() -> None:
    """Verify transitions accumulation and mutual information calculation."""
    # Mock sequence mapping
    sid2seq = {
        "sid1": "ABC",
        "sid2": "BCD",
    }
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    # Write a temporary pairfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        # Match alignment of 3 residues using P for Perfect Match
        f.write("sid1 sid2 3P\n")
        pairfile_path = f.name

    try:
        counts, counts_prev = accumulate_counts(pairfile_path, sid2seq, letter2idx, n_letters=4)

        # 3P matches: [0, 0] (A-B), [1, 1] (B-C), [2, 2] (C-D)
        # Bidirectional pairs:
        # A(0) <-> B(1) -> counts[0, 1] += 1, counts[1, 0] += 1
        # B(1) <-> C(2) -> counts[1, 2] += 1, counts[2, 1] += 1
        # C(2) <-> D(3) -> counts[2, 3] += 1, counts[3, 2] += 1
        assert counts[0, 1] == 1
        assert counts[1, 0] == 1
        assert counts[1, 2] == 1
        assert counts[2, 3] == 1
        assert counts.sum() == 6

        # Lagged transition counts:
        # For k=1 (i=1, j=1):
        # target-lag: counts_prev[seq1[1], seq2[0]] = counts_prev[B, B] = counts_prev[1, 1] += 1
        # source-lag: counts_prev[seq2[1], seq1[0]] = counts_prev[C, A] = counts_prev[2, 0] += 1
        # For k=2 (i=2, j=2):
        # target-lag: counts_prev[seq1[2], seq2[1]] = counts_prev[C, C] = counts_prev[2, 2] += 1
        # source-lag: counts_prev[seq2[2], seq1[1]] = counts_prev[D, B] = counts_prev[3, 1] += 1
        assert counts_prev[1, 1] == 1
        assert counts_prev[2, 0] == 1
        assert counts_prev[2, 2] == 1
        assert counts_prev[3, 1] == 1
        assert counts_prev.sum() == 4

        # Calculate mutual information
        mi, mi_tot = calc_alphabet_mi(counts + 1, counts_prev + 1)
        assert mi > 0.0
        assert mi_tot is not None

    finally:
        Path(pairfile_path).unlink()


def test_merge_columns_counts_preservation() -> None:
    """Verify merge_columns consolidates matrix elements preserving summation."""
    counts = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    # Merge row/col index 2 into index 1
    new_counts = merge_columns(counts, i=2, j=1)

    assert new_counts.shape == (2, 2)
    # Sum of counts must remain exactly conserved
    assert new_counts.sum() == counts.sum()


# =====================================================================
# 5. Inference and Predict Fallback Tests
# =====================================================================


def test_inference_fallback_discretize() -> None:
    """Test predict and discretize fallback options and exclusions."""
    # Mock bare PyTorch model (nn.Linear)
    encoder = nn.Linear(10, 4)
    x = np.random.randn(8, 10).astype(np.float32)

    # 1. Normal predict
    z = predict(encoder, x)
    assert z.shape == (8, 4)

    # 2. Discretize manual fallback path using centroids
    centroids = np.random.randn(5, 4).astype(np.float32)
    indices = discretize(encoder, centroids, x)
    assert indices.shape == (8,)
    assert np.all(indices >= 0) and np.all(indices < 5)

    # 3. Discretize fallback raises error if centroids is None
    with pytest.raises(ValueError, match="Centroids must be provided"):
        _ = discretize(encoder, None, x)
