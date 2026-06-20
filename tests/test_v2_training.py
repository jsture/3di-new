"""Tests for the single-path v2 model, quantizers, data utilities, and training loop."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tdi.v2 import (
    AlphabetModel,
    EMAVectorQuantizer,
    FSQQuantizer,
    PairDataset,
    ResidualMLP,
    make_quantizer,
)
from tdi.v2.train import train_model
from tdi.v2.train_config import (
    DataConfig,
    LoopConfig,
    ModelConfig,
    OutputsConfig,
    TrainConfig,
)
from tdi.v2.training_data import (
    _superposed_ca_distances,
    filter_ca_distance,
    filter_valid_pairs,
    make_bidirectional_pairs,
    parse_alignment,
)
from tdi.v2.util import parse_cigar

# ---------------------------------------------------------------------------
# Quantizers (shared (z_q, indices, q_loss, metrics) interface)
# ---------------------------------------------------------------------------


def test_residual_mlp_shapes() -> None:
    """ResidualMLP outputs the requested dimensions."""
    mlp = ResidualMLP(input_dim=10, hidden_dim=64, output_dim=4, depth=3)
    out = mlp(torch.randn(8, 10))
    assert out.shape == (8, 4)


def test_ema_quantizer_interface() -> None:
    """EMA-VQ returns (z_q, indices, q_loss, metrics) with the VQ metric set."""
    quantizer = EMAVectorQuantizer(n_states=20, z_dim=4, l2_normalize=True)
    z = torch.randn(8, 4)
    z_q, indices, q_loss, metrics = quantizer(z)

    assert z_q.shape == (8, 4)
    assert indices.shape == (8,)
    assert q_loss.shape == ()
    assert set(metrics) == {"perplexity", "n_replaced", "margin"}


def test_fsq_quantizer_interface() -> None:
    """FSQ returns q_loss == 0 and only the perplexity metric (no VQ margin)."""
    quantizer = FSQQuantizer(levels=[5, 4])
    z = torch.randn(8, 2)
    z_q, indices, q_loss, metrics = quantizer(z)

    assert z_q.shape == (8, 2)
    assert indices.shape == (8,)
    assert float(q_loss) == 0.0
    assert set(metrics) == {"perplexity"}


def test_fsq_quantizer_grids() -> None:
    """FSQ grid creation for odd vs even levels."""
    q_odd = FSQQuantizer(levels=[3])
    assert torch.allclose(
        q_odd.implicit_codebook, torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    )
    q_even = FSQQuantizer(levels=[4])
    assert torch.allclose(
        q_even.implicit_codebook,
        torch.tensor([[-0.75], [-0.25], [0.25], [0.75]], dtype=torch.float32),
    )


def test_vq_dead_code_replacement() -> None:
    """Dead VQ centroids are replaced with batch latents after warmup."""
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
    quantizer.embedding.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0], [10.0, 10.0]]))
    quantizer.ema_sum.copy_(quantizer.embedding)
    quantizer.ema_count.copy_(torch.tensor([1.0, 1.0, 1.0]))

    z1 = torch.tensor([[0.9, 0.9], [-0.9, -0.9]], dtype=torch.float32)
    assert float(quantizer(z1)[3]["n_replaced"]) == 0.0
    assert float(quantizer(z1)[3]["n_replaced"]) == 0.0

    z2 = torch.tensor([[5.0, -5.0], [5.0, -5.0]], dtype=torch.float32)
    _, _, _, metrics = quantizer(z2)
    assert float(metrics["n_replaced"]) > 0
    # With l2_normalize the replaced code is renormalized, so it aligns with [5, -5]'s direction.
    expected = torch.nn.functional.normalize(torch.tensor([[5.0, -5.0]]), dim=-1)[0]
    assert torch.allclose(quantizer.embedding[2], expected, atol=1e-4)


def test_make_quantizer_factory() -> None:
    """The factory dispatches by name and rejects unknown quantizers."""
    assert isinstance(make_quantizer("vq", n_states=20, z_dim=4), EMAVectorQuantizer)
    assert isinstance(make_quantizer("fsq", n_states=20, z_dim=2, levels=[5, 4]), FSQQuantizer)
    with pytest.raises(ValueError, match="Unknown quantizer"):
        make_quantizer("nope", n_states=20, z_dim=4)


# ---------------------------------------------------------------------------
# AlphabetModel
# ---------------------------------------------------------------------------


def test_model_forward_returns_contract() -> None:
    """forward yields y_hat / indices / q_loss / metrics."""
    model = AlphabetModel()
    out = model(torch.randn(8, 10))
    assert out["y_hat"].shape == (8, 10)
    assert out["indices"].shape == (8,)
    assert out["q_loss"].shape == ()
    assert "perplexity" in out["metrics"]


def test_n_states_cap_raises() -> None:
    """A codebook larger than the alphabet is rejected at construction."""
    with pytest.raises(ValueError, match="exceeds alphabet size"):
        AlphabetModel(quantizer="vq", n_states=51)


@pytest.mark.parametrize(
    ("quantizer", "kwargs"),
    [("vq", {"n_states": 24}), ("fsq", {"levels": [5, 5]})],
)
def test_model_trains_one_step(quantizer: str, kwargs: dict) -> None:
    """Both quantizers train one step and reduce a tiny-batch loss."""
    model = AlphabetModel(quantizer=quantizer, **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(16, 10)
    y = torch.randn(16, 10)

    out = model(x)
    loss = torch.nn.functional.smooth_l1_loss(out["y_hat"], y) + out["q_loss"]
    assert loss.ndim == 0
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_fp32_forward_is_finite() -> None:
    """fp32 forward yields finite y_hat and q_loss for VQ."""
    model = AlphabetModel(quantizer="vq")
    out = model(torch.randn(8, 10))
    assert torch.isfinite(out["y_hat"]).all()
    assert torch.isfinite(out["q_loss"])


def test_encode_states_does_not_change_mode() -> None:
    """encode_states leaves the model in its prior train/eval mode."""
    model = AlphabetModel()
    model.train()
    _ = model.encode_states(torch.randn(4, 10))
    assert model.training is True


def test_forward_is_differentiable() -> None:
    """The reconstruction path flows gradient to the input and the decoder."""
    model = AlphabetModel()
    x = torch.randn(16, 10, requires_grad=True)
    y = torch.randn(16, 10)
    out = model(x)
    loss = torch.nn.functional.smooth_l1_loss(out["y_hat"], y) + out["q_loss"]
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert model.decoder.output[-1].weight.grad is not None


def test_tiny_batch_overfit() -> None:
    """The model can drive down a fixed tiny-batch loss."""
    model = AlphabetModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    def step() -> float:
        with torch.no_grad():
            out = model(x)
            return float(torch.nn.functional.smooth_l1_loss(out["y_hat"], y) + out["q_loss"])

    initial = step()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.smooth_l1_loss(out["y_hat"], y) + out["q_loss"]
        loss.backward()
        optimizer.step()
    assert step() < initial


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_export_roundtrip_states_identical(tmp_path: Path) -> None:
    """A round-tripped VQ export assigns identical states."""
    model = AlphabetModel()
    model.eval()
    x = torch.randn(16, 10)
    states_before = model.encode_states(x)

    model.save(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded, _, _ = AlphabetModel.load(tmp_path)
    loaded.eval()
    assert torch.equal(states_before, loaded.encode_states(x))


def test_export_roundtrip_restores_decoder(tmp_path: Path) -> None:
    """The decoder is exported and a loaded model reconstructs a finite y_hat."""
    model = AlphabetModel()
    model.save(tmp_path, mean=np.zeros(10), std=np.ones(10))
    assert (tmp_path / "decoder_state_dict.pt").exists()

    loaded, _, _ = AlphabetModel.load(tmp_path)
    with torch.no_grad():
        out = loaded(torch.randn(4, 10))
    assert out["y_hat"].shape == (4, 10)
    assert torch.isfinite(out["y_hat"]).all()


def test_fsq_export_roundtrip_preserves_levels(tmp_path: Path) -> None:
    """FSQ levels round-trip and the correct artifacts are written."""
    model = AlphabetModel(quantizer="fsq", levels=[5, 4])
    model.save(tmp_path, mean=np.zeros(10), std=np.ones(10))
    assert (tmp_path / "fsq_levels.json").exists()
    assert not (tmp_path / "centroids.npy").exists()

    loaded, _, _ = AlphabetModel.load(tmp_path)
    assert loaded.levels == [5, 4]
    assert loaded.z_dim == 2


def test_export_preserves_scaler(tmp_path: Path) -> None:
    """Scaler mean/std round-trip through the export."""
    model = AlphabetModel()
    mean = np.random.randn(10).astype(np.float32)
    std = (np.random.rand(10) + 0.1).astype(np.float32)
    model.save(tmp_path, mean=mean, std=std)
    _loaded, loaded_mean, loaded_std = AlphabetModel.load(tmp_path)
    assert np.allclose(mean, loaded_mean)
    assert np.allclose(std, loaded_std)


# ---------------------------------------------------------------------------
# k-means codebook init
# ---------------------------------------------------------------------------


def _tiny_loader(n: int = 200, batch_size: int = 10):
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(TensorDataset(torch.randn(n, 10), torch.randn(n, 10)), batch_size=batch_size)


def test_kmeans_init_populates_codebook() -> None:
    """k-means init replaces the random codebook and marks it initialized."""
    model = AlphabetModel(quantizer="vq", n_states=20, z_dim=4)
    quantizer = model.quantizer
    assert isinstance(quantizer, EMAVectorQuantizer)
    before = quantizer.embedding.clone()
    assert bool(quantizer.initialized.item()) is False

    model.init_codebook_from_loader(_tiny_loader(), n_batches=4)

    assert bool(quantizer.initialized.item()) is True
    assert not torch.equal(before, quantizer.embedding)


def test_kmeans_init_noop_for_fsq() -> None:
    """k-means init is a no-op for the FSQ backend."""
    model = AlphabetModel(quantizer="fsq", levels=[5, 4])
    model.init_codebook_from_loader(_tiny_loader())
    assert isinstance(model.quantizer, FSQQuantizer)


# ---------------------------------------------------------------------------
# PairDataset
# ---------------------------------------------------------------------------


def test_pair_dataset_scaling() -> None:
    """PairDataset fits a scaler and yields (x, y) tensors without RNG state."""
    x = np.random.randn(10, 10) * 5.0 + 3.0
    y = np.random.randn(10, 10) * 2.0 - 1.0

    dataset = PairDataset(x, y)
    assert dataset.mean.shape == (10,)
    assert not hasattr(dataset, "set_epoch")

    x_scaled, y_scaled = dataset[0]
    assert x_scaled.shape == (10,)
    assert y_scaled.shape == (10,)


# ---------------------------------------------------------------------------
# Data-path utilities (unchanged through the refactor)
# ---------------------------------------------------------------------------


def _manual_superposed_ca_distances(ca_fixed: np.ndarray, ca_moving: np.ndarray) -> np.ndarray:
    """Reference implementation matching the previous handwritten SVD path."""
    fixed_center = ca_fixed.mean(axis=0)
    moving_center = ca_moving.mean(axis=0)
    fixed0 = ca_fixed - fixed_center
    moving0 = ca_moving - moving_center

    h = moving0.T @ fixed0
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T

    moving_aligned = (rot @ ca_moving.T).T + fixed_center - (rot @ moving_center.T).T
    return np.linalg.norm(ca_fixed - moving_aligned, axis=1).astype(np.float32)


def test_ca_filter_requires_superposition() -> None:
    """Ca filtering superposes translated and rotated coordinates."""
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rot_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    coords2 = coords1 @ rot_z.T + np.array([100.0, 50.0, -20.0])

    idx = np.array([0, 1, 2])
    v1, v2, dists, error = filter_ca_distance(idx, idx, coords1, coords2, max_ca_dist=1.0)
    assert error is None
    assert np.array_equal(v1, idx) and np.array_equal(v2, idx)
    assert dists is not None and np.all(dists < 1e-5)


def test_superposed_distances_match_manual_svd_reference() -> None:
    """SciPy rotation matches the previous handwritten SVD implementation."""
    rng = np.random.default_rng(123)
    coords1 = rng.normal(size=(8, 3))
    rot_z = np.array(
        [[0.5, -np.sqrt(3) / 2, 0.0], [np.sqrt(3) / 2, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    coords2 = coords1 @ rot_z.T + np.array([3.0, -7.0, 2.0])
    assert np.allclose(
        _superposed_ca_distances(coords1, coords2),
        _manual_superposed_ca_distances(coords1, coords2),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "coords",
    [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ],
)
def test_ca_filter_rejects_rank_deficient_coordinates(coords: np.ndarray) -> None:
    """Degenerate superposition inputs retain the existing error label."""
    idx = np.array([0, 1, 2])
    v1, v2, dists, error = filter_ca_distance(idx, idx, coords, coords)
    assert error == "rank_deficient_coordinates"
    assert len(v1) == 0 and len(v2) == 0
    assert dists is not None and len(dists) == 0


def test_ca_filter_drops_too_few_pairs() -> None:
    """Too-few-pair inputs are dropped instead of emitted with NaN distance."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    idx = np.array([0, 1])
    v1, v2, dists, error = filter_ca_distance(idx, idx, coords, coords)
    assert error == "too_few_pairs"
    assert len(v1) == 0 and len(v2) == 0
    assert dists is not None and len(dists) == 0


def test_parse_cigar_empty_shape() -> None:
    """CIGAR parsing with no P ops yields an empty (0, 2) pair array."""
    assert parse_cigar("10M5I3D").shape == (0, 2)


def test_parse_alignment() -> None:
    """parse_alignment returns the matched alignment pairs."""
    assert parse_alignment("3P5M").shape == (3, 2)


def test_filter_valid_pairs() -> None:
    """Alignment filtering drops residue positions invalid in either structure."""
    idx_1 = np.array([0, 1, 2])
    idx_2 = np.array([0, 1, 2])
    mask1 = np.array([True, False, True])
    mask2 = np.array([True, True, False])
    v1, v2 = filter_valid_pairs(idx_1, idx_2, mask1, mask2)
    assert np.array_equal(v1, np.array([0]))
    assert np.array_equal(v2, np.array([0]))


def test_make_bidirectional_pairs() -> None:
    """Bidirectional pairs double the rows (forward then reverse)."""
    feat1 = np.ones((5, 10))
    feat2 = np.ones((5, 10)) * 2
    x, y = make_bidirectional_pairs(feat1, feat2, np.array([0, 1]), np.array([1, 2]))
    assert x.shape == (4, 10)
    assert y.shape == (4, 10)


def test_deterministic_rng_capping() -> None:
    """Alignment-specific hashing and rng choice are reproducible."""
    import hashlib

    hasher = hashlib.sha256(f"{'d1qksa1-d1gwua_'}:{123}".encode())
    cap_seed = int(hasher.hexdigest(), 16) % (2**32)
    rng1 = np.random.default_rng(cap_seed)
    rng2 = np.random.default_rng(cap_seed)
    assert np.array_equal(rng1.choice(100, 10, replace=False), rng2.choice(100, 10, replace=False))


def test_scop_grouping_logic(tmp_path: Path) -> None:
    """make_splits.py groups by SCOP superfamily."""
    lookup_file = tmp_path / "scop_lookup.tsv"
    lookup_file.write_text("d1qksa1\ta.3.1.2\nd1gwua_\ta.3.1.5\nd1i17a_\tb.1.1.1\n")
    pdbs_file = tmp_path / "pdbs.txt"
    pdbs_file.write_text("d1qksa1\nd1gwua_\nd1i17a_\nd_fallback\n")

    import subprocess

    import pandas as pd

    out_dir = tmp_path / "splits"
    res = subprocess.run(
        [
            "python3",
            "scripts/make_splits.py",
            str(pdbs_file),
            str(out_dir),
            "--scop_lookup",
            str(lookup_file),
            "--group_by",
            "superfamily",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    df = pd.read_csv(out_dir / "train_manifest.csv")
    assert df[df["structure_id"] == "d1qksa1"].iloc[0]["group_id"] == "a.3.1"
    assert df[df["structure_id"] == "d1gwua_"].iloc[0]["group_id"] == "a.3.1"


# ---------------------------------------------------------------------------
# End-to-end training loop
# ---------------------------------------------------------------------------


def _write_processed_dir(path: Path, n: int = 64, dim: int = 10) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        np.save(path / f"{split}_x_raw.npy", rng.standard_normal((n, dim)).astype(np.float32))
        np.save(path / f"{split}_y_raw.npy", rng.standard_normal((n, dim)).astype(np.float32))
    np.savez(
        path / "scaler.npz",
        mean=np.zeros(dim, dtype=np.float32),
        std=np.ones(dim, dtype=np.float32),
    )


def test_train_model_end_to_end_writes_export(tmp_path: Path) -> None:
    """A short VQ run trains without Lightning and writes the self-describing export."""
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_processed_dir(processed)

    cfg = TrainConfig(
        model=ModelConfig(quantizer="vq", n_states=16, z_dim=4),
        train=LoopConfig(batch_size=16, max_epochs=2, kmeans_init=True, kmeans_init_batches=2),
        data=DataConfig(processed_dir=str(processed)),
        outputs=OutputsConfig(out_dir=str(tmp_path / "run")),
    )
    model = train_model(cfg)
    assert isinstance(model, AlphabetModel)

    out_dir = tmp_path / "run"
    for name in (
        "config.json",
        "encoder_state_dict.pt",
        "decoder_state_dict.pt",
        "scaler.json",
        "centroids.npy",
        "train_log.csv",
        "run_config.resolved.json",
    ):
        assert (out_dir / name).exists(), name

    loaded, _, _ = AlphabetModel.load(out_dir)
    assert loaded.n_states == 16
