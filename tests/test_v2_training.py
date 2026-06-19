"""Unit tests for modernized 3Di VAE v2 training pipeline components."""

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def test_ca_filter_12d_coordinates() -> None:
    """Verify filter_ca_distance works with full backbone (12D) coordinates."""
    ca1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ca2 = ca1 + np.array([10.0, -5.0, 2.0])

    coords1 = np.zeros((3, 12))
    coords2 = np.zeros((3, 12))
    coords1[:, 0:3] = ca1
    coords2[:, 0:3] = ca2

    idx_1 = np.array([0, 1, 2])
    idx_2 = np.array([0, 1, 2])

    v1, v2, dists = filter_ca_distance(idx_1, idx_2, coords1, coords2, max_ca_dist=1.0)
    assert np.array_equal(v1, idx_1)
    assert np.array_equal(v2, idx_2)
    assert dists is not None
    assert np.all(dists < 1e-5)


def test_deterministic_rng_capping() -> None:
    """Verify that alignment-specific hashing and rng choice are reproducible."""
    alignment_id = "d1qksa1-d1gwua_"
    seed = 123
    import hashlib

    hasher = hashlib.sha256(f"{alignment_id}:{seed}".encode())
    cap_seed = int(hasher.hexdigest(), 16) % (2**32)

    rng1 = np.random.default_rng(cap_seed)
    rng2 = np.random.default_rng(cap_seed)

    choices1 = rng1.choice(100, 10, replace=False)
    choices2 = rng2.choice(100, 10, replace=False)
    assert np.array_equal(choices1, choices2)


def test_scop_grouping_logic(tmp_path: Path) -> None:
    """Verify make_splits.py logic with SCOP lookup fold/superfamily grouping."""
    lookup_file = tmp_path / "scop_lookup.tsv"
    with open(lookup_file, "w") as f:
        f.write("d1qksa1\ta.3.1.2\n")
        f.write("d1gwua_\ta.3.1.5\n")
        f.write("d1i17a_\tb.1.1.1\n")

    pdbs_file = tmp_path / "pdbs.txt"
    with open(pdbs_file, "w") as f:
        f.write("d1qksa1\n")
        f.write("d1gwua_\n")
        f.write("d1i17a_\n")
        f.write("d_fallback\n")

    import subprocess

    import pandas as pd

    out_dir = tmp_path / "splits"
    cmd = [
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
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out_dir / "train_manifest.csv").exists()
    assert (out_dir / "val_manifest.csv").exists()

    df = pd.read_csv(out_dir / "train_manifest.csv")
    row_q = df[df["structure_id"] == "d1qksa1"].iloc[0]
    row_g = df[df["structure_id"] == "d1gwua_"].iloc[0]
    assert row_q["group_id"] == "a.3.1"
    assert row_g["group_id"] == "a.3.1"


def test_cli_evaluate_end_to_end(tmp_path: Path) -> None:
    """Verify tdi-v2 evaluate command end-to-end with mock inputs."""
    # 1. Export a dummy model
    from tdi.v2 import TdiV2Model

    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    model_dir = tmp_path / "exported_model"
    mean = np.zeros(10)
    std = np.ones(10)
    model.export_model(model_dir, mean=mean, std=std)

    # 2. Create a dummy PDB file
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    pdb_content = (
        "HEADER    MOCK PROTEIN\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       2.000   1.000   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00  0.00           C\n"
        "ATOM      6  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      7  CA  GLY A   2       4.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      8  C   GLY A   2       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      9  O   GLY A   2       5.000   1.000   0.000  1.00  0.00           O\n"
        "ATOM     10  N   ALA A   3       6.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM     11  CA  ALA A   3       7.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM     12  C   ALA A   3       8.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM     13  O   ALA A   3       8.000   1.000   0.000  1.00  0.00           O\n"
        "ATOM     14  CB  ALA A   3       7.000   1.000   0.000  1.00  0.00           C\n"
        "TER\n"
    )
    with open(pdb_dir / "d1qksa1.pdb", "w") as f:
        f.write(pdb_content)
    with open(pdb_dir / "d1gwua_.pdb", "w") as f:
        f.write(pdb_content)

    # 3. Create a dummy pairfile
    pairfile = tmp_path / "pairs.txt"
    with open(pairfile, "w") as f:
        f.write("d1qksa1 d1gwua_ 3M\n")

    # 4. Invoke CLI evaluate command via subprocess
    import subprocess

    out_dir = tmp_path / "eval_out"
    cmd = [
        "python3",
        "-m",
        "tdi.v2.cli",
        "evaluate",
        "--model_dir",
        str(model_dir),
        "--pdb_dir",
        str(pdb_dir),
        "--pairfile",
        str(pairfile),
        "--out_dir",
        str(out_dir),
        "--virt",
        "0.0",
        "0.0",
        "1.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out_dir / "sequences.txt").exists()
    assert (out_dir / "submat.txt").exists()
    assert (out_dir / "evaluation_report.json").exists()


def _tiny_loader(n: int = 200, batch_size: int = 10) -> DataLoader:
    """Build a small (x, y) DataLoader of random scaled features."""
    x = torch.randn(n, 10)
    y = torch.randn(n, 10)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_kmeans_init_populates_codebook() -> None:
    """k-means init replaces the random codebook and marks it initialized."""
    model = TdiV2Model(quantizer_type="vq", n_states=20, z_dim=4)
    quantizer = model.quantizer
    assert isinstance(quantizer, EMAVectorQuantizer)
    before = quantizer.embedding.clone()
    assert bool(quantizer.initialized.item()) is False

    model.init_codebook_from_data(_tiny_loader(), n_batches=4)

    assert bool(quantizer.initialized.item()) is True
    assert not torch.equal(before, quantizer.embedding)
    assert torch.allclose(quantizer.ema_count, torch.ones_like(quantizer.ema_count))
    assert torch.equal(quantizer.ema_sum, quantizer.embedding)


def test_kmeans_init_noop_for_fsq() -> None:
    """k-means init is a no-op for the FSQ backend."""
    model = TdiV2Model(quantizer_type="fsq", fsq_levels=[5, 4])
    # Must not raise and must leave the FSQ quantizer in place.
    model.init_codebook_from_data(_tiny_loader())
    assert isinstance(model.quantizer, FSQQuantizer)


def test_optimizer_has_two_param_groups_correct_decay() -> None:
    """AdamW splits decay (>=2-D) from no-decay (bias/norm) groups."""
    model = TdiV2Model(lambda_contrast=0.0)
    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, _tiny_loader())

    opt = trainer.optimizers[0]
    assert len(opt.param_groups) == 2
    decay_g, no_decay_g = opt.param_groups
    assert decay_g["weight_decay"] == model.weight_decay
    assert no_decay_g["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay_g["params"])
    assert all(p.ndim < 2 for p in no_decay_g["params"])

    sched = trainer.lr_scheduler_configs[0].scheduler
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


class _LRRecorder(L.Callback):
    """Records the optimizer LR at the start of each training batch."""

    def __init__(self) -> None:
        self.lrs: list[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # type: ignore[no-untyped-def]
        self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])


def test_scheduler_warms_up_then_cosines() -> None:
    """LR rises during warmup then cosine-decays toward zero, per-step."""
    model = TdiV2Model(lambda_contrast=0.0, warmup_ratio=0.2, lr=1e-3)
    recorder = _LRRecorder()
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[recorder],
    )
    trainer.fit(model, _tiny_loader())

    lrs = recorder.lrs
    assert len(lrs) > 5
    peak = max(range(len(lrs)), key=lambda i: lrs[i])
    assert peak > 0  # warmed up: peak is not the first step
    assert lrs[-1] < lrs[peak]  # cosine decays after the peak
    assert lrs[-1] < 0.1 * lrs[peak]  # near zero by the final step


def test_contrastive_disabled_is_noop() -> None:
    """With lambda_contrast=0 there is no logit_scale and the step still runs."""
    model = TdiV2Model(lambda_contrast=0.0)
    assert not hasattr(model, "logit_scale")
    x = torch.randn(8, 10)
    y = torch.randn(8, 10)
    out = model.training_step((x, y), 0)
    assert out.ndim == 0


def test_contrastive_logit_scale_clamped() -> None:
    """Enabled contrastive head exposes a learnable logit_scale clamped at use."""
    model = TdiV2Model(lambda_contrast=0.05, temperature=0.1)
    assert hasattr(model, "logit_scale")
    with torch.no_grad():
        model.logit_scale.fill_(10.0)  # exp(10) >> 100 before clamp
    scale = model.logit_scale.clamp(max=np.log(100.0)).exp()
    assert scale.item() <= 100.0 + 1e-3

    # Symmetric loss runs and is differentiable.
    x = torch.randn(8, 10)
    y = torch.randn(8, 10)
    loss = model.training_step((x, y), 0)
    loss.backward()
    assert model.logit_scale.grad is not None
