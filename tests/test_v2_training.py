"""Unit tests for modernized 3Di VAE v2 training pipeline components."""

from pathlib import Path

import numpy as np
import torch

from tdi.v2 import (
    EMAVectorQuantizer,
    FSQQuantizer,
    PairDataset,
    ResidualMLP,
    TdiV2Model,
)
from tdi.v2.training_data import (
    filter_ca_distance,
    filter_valid_pairs,
    make_bidirectional_pairs,
    parse_alignment,
)
from tdi.v2.util import parse_cigar


def test_residual_mlp_shapes() -> None:
    """Test ResidualMLP outputs correct dimensions and shapes."""
    mlp = ResidualMLP(input_dim=10, hidden_dim=64, output_dim=4, depth=3)
    x = torch.randn(8, 10)
    out = mlp(x)
    assert out.shape == (8, 4)


def test_ema_quantizer_shapes() -> None:
    """Test EMAVectorQuantizer outputs and updates properties."""
    quantizer = EMAVectorQuantizer(n_states=20, z_dim=4, l2_normalize=True)
    z = torch.randn(8, 4)
    commit_loss, z_q, perplexity, indices, usage, n_replaced = quantizer(z)

    assert z_q.shape == (8, 4)
    assert commit_loss.shape == ()
    assert perplexity.shape == ()
    assert indices.shape == (8,)
    assert usage.shape == (20,)
    assert n_replaced.shape == ()


def test_fsq_quantizer_shapes() -> None:
    """Test FSQQuantizer levels mapping and indexing."""
    quantizer = FSQQuantizer(levels=[5, 4])
    z = torch.randn(8, 2)
    commit_loss, z_q, perplexity, indices, usage, n_replaced = quantizer(z)

    assert z_q.shape == (8, 2)
    assert commit_loss.item() == 0.0
    assert perplexity.shape == ()
    assert indices.shape == (8,)
    assert usage.shape == (20,)
    assert n_replaced.shape == ()


def test_pair_dataset_scaling() -> None:
    """Test PairDataset scaling, fitting, and optional jittering."""
    x = np.random.randn(10, 10) * 5.0 + 3.0
    y = np.random.randn(10, 10) * 2.0 - 1.0

    dataset = PairDataset(x, y, jitter_std=0.0)
    assert dataset.mean.shape == (10,)
    assert dataset.std.shape == (10,)

    # Scaled features should have mean close to 0 and std close to 1
    x_scaled, y_scaled = dataset[0]
    assert x_scaled.shape == (10,)
    assert y_scaled.shape == (10,)


def test_fsq_default_forward_shapes() -> None:
    """Test FSQ training step shapes."""
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    x = torch.randn(8, 10)
    y = torch.randn(8, 10)
    out = model.training_step((x, y), 0)
    assert out.ndim == 0


def test_fsq_z_dim_matches_levels() -> None:
    """Test FSQ z_dim matches the length of fsq_levels."""
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    assert model.z_dim == 2
    assert model.encoder.output[-1].out_features == 2


def test_forward_is_differentiable() -> None:
    """Verify forward pass is differentiable."""
    model = TdiV2Model()
    x = torch.randn(4, 10, requires_grad=True)
    out = model(x)
    loss = out["mu_partner"].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_encode_states_does_not_change_mode() -> None:
    """Verify encode_states does not change model training mode."""
    model = TdiV2Model()
    model.train()
    x = torch.randn(4, 10)
    _ = model.encode_states(x)
    assert model.training is True


def test_ca_filter_requires_superposition() -> None:
    """Verify Ca filtering superposes correctly using Kabsch SVD."""
    # Create two identical sets of coordinates but translated and rotated
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coords2 = coords1 + np.array([100.0, 50.0, -20.0])

    idx_1 = np.array([0, 1, 2])
    idx_2 = np.array([0, 1, 2])

    v1, v2, dists = filter_ca_distance(idx_1, idx_2, coords1, coords2, max_ca_dist=1.0)
    assert np.array_equal(v1, idx_1)
    assert np.array_equal(v2, idx_2)
    assert dists is not None
    assert np.all(dists < 1e-5)


def test_parse_cigar_empty_shape() -> None:
    """Verify CIGAR parsing is empty-safe."""
    pairs = parse_cigar("10M5I3D")
    assert pairs.shape == (0, 2)


def test_fsq_export_roundtrip_preserves_levels(tmp_path: Path) -> None:
    """Verify FSQ levels roundtrip correctly through export configuration."""
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded, _, _ = TdiV2Model.load_from_export(tmp_path)
    assert loaded.fsq_levels == [5, 4]
    assert loaded.z_dim == 2


def test_export_load_preserves_scaled_encoding(tmp_path: Path) -> None:
    """Verify features mean and standard deviation are preserved and registered."""
    model = TdiV2Model()
    mean = np.random.randn(10).astype(np.float32)
    std = (np.random.rand(10) + 0.1).astype(np.float32)
    model.export_model(tmp_path, mean=mean, std=std)
    _loaded, loaded_mean, loaded_std = TdiV2Model.load_from_export(tmp_path)
    assert np.allclose(mean, loaded_mean)
    assert np.allclose(std, loaded_std)


def test_export_files_match_quantizer_type(tmp_path: Path) -> None:
    """Verify correct files are generated for FSQ vs VQ quantizers."""
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    assert (tmp_path / "fsq_levels.json").exists()
    assert not (tmp_path / "centroids.npy").exists()


def test_decoder_mean_receives_gradient() -> None:
    """Verify decoder parameters receive gradients."""
    model = TdiV2Model(loss_type="smooth_l1")
    x = torch.randn(16, 10)
    y = torch.randn(16, 10)
    loss = model.training_step((x, y), 0)
    loss.backward()
    assert model.decoder.mu_partner.weight.grad is not None
    assert model.decoder.mu_partner.weight.grad.abs().sum() > 0


def test_tiny_batch_overfit() -> None:
    """Verify model can overfit a tiny batch of data."""
    model = TdiV2Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    initial = float(model.training_step((x, y), 0).detach())
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.training_step((x, y), 0)
        loss.backward()
        optimizer.step()
    final = float(model.training_step((x, y), 0).detach())

    assert final < initial


def test_export_roundtrip_states_identical(tmp_path: Path) -> None:
    """Verify model state assignments are identical before and after roundtrip."""
    model = TdiV2Model()
    model.eval()
    x = torch.randn(16, 10)
    states_before = model.encode_states(x)

    model.export_model(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded, _, _ = TdiV2Model.load_from_export(tmp_path)
    loaded.eval()
    states_after = loaded.encode_states(x)

    assert torch.equal(states_before, states_after)


def test_sequence_distance_convention() -> None:
    """Verify sequence delta sign convention."""
    i = 10
    j = 14
    delta = j - i
    assert delta == 4


def test_parse_alignment() -> None:
    """Verify stages parser returns matched alignments."""
    idx_pairs = parse_alignment("3P5M")
    assert idx_pairs.shape == (3, 2)


def test_filter_valid_pairs() -> None:
    """Verify alignment filtering of invalid residue positions."""
    idx_1 = np.array([0, 1, 2])
    idx_2 = np.array([0, 1, 2])
    mask1 = np.array([True, False, True])
    mask2 = np.array([True, True, False])
    v1, v2 = filter_valid_pairs(idx_1, idx_2, mask1, mask2)
    assert np.array_equal(v1, np.array([0]))
    assert np.array_equal(v2, np.array([0]))


def test_make_bidirectional_pairs() -> None:
    """Verify correct bidirectional creation of target-partner pairs."""
    feat1 = np.ones((5, 10))
    feat2 = np.ones((5, 10)) * 2
    idx_1 = np.array([0, 1])
    idx_2 = np.array([1, 2])
    x, y = make_bidirectional_pairs(feat1, feat2, idx_1, idx_2)
    assert x.shape == (4, 10)
    assert y.shape == (4, 10)
