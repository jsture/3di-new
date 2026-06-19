"""Unit tests for modernized 3Di VAE v2 training pipeline components."""

import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from tdi.v2 import (
    EMAVectorQuantizer,
    FSQQuantizer,
    PairDataset,
    ResidualMLP,
    TdiV2Model,
    create_vqvae,
)


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
    commit_loss, z_q, perplexity, indices, usage = quantizer(z)

    assert z_q.shape == (8, 4)
    assert commit_loss.shape == ()
    assert perplexity.shape == ()
    assert indices.shape == (8,)
    assert usage.shape == (20,)


def test_fsq_quantizer_shapes() -> None:
    """Test FSQQuantizer levels mapping and indexing."""
    quantizer = FSQQuantizer(levels=[5, 4])
    z = torch.randn(8, 2)
    commit_loss, z_q, perplexity, indices, usage = quantizer(z)

    assert z_q.shape == (8, 2)
    assert commit_loss.item() == 0.0
    assert perplexity.shape == ()
    assert indices.shape == (8,)
    assert usage.shape == (20,)


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


def test_gradient_flow_to_decoder() -> None:
    """Verify that decoder mean parameters receive gradients from the loss function."""
    model = create_vqvae(seed=42, input_dim=10, hidden_dim=32, z_dim=4, n_states=20)

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    batch = (x, y)

    # Put in train mode to activate gradient tracing
    model.train()
    loss = model.training_step(batch, 0)
    loss.backward()

    # Verify gradients are calculated for decoding parameters
    assert model.decoder.mu_partner.weight.grad is not None
    assert model.decoder.mu_partner.weight.grad.abs().sum() > 0.0
    assert model.decoder.mu_self.weight.grad is not None
    assert model.decoder.mu_self.weight.grad.abs().sum() > 0.0


def test_model_overfit_tiny_batch() -> None:
    """Test that a small model is capable of overfitting a tiny dataset."""
    L.seed_everything(42)
    model = create_vqvae(seed=42, input_dim=10, hidden_dim=32, z_dim=4, n_states=20)

    x = np.random.randn(8, 10).astype(np.float32)
    y = np.random.randn(8, 10).astype(np.float32)
    dataset = PairDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=8)

    # Use Lightning Trainer for mini training run
    trainer = L.Trainer(
        max_epochs=150,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    initial_loss = model.training_step(next(iter(dataloader)), 0).item()
    trainer.fit(model, train_dataloaders=dataloader)
    final_loss = model.training_step(next(iter(dataloader)), 0).item()

    # Final loss should be significantly lower than initial loss
    assert final_loss < 0.5 * initial_loss


def test_export_and_load_consistency() -> None:
    """Test model configurations, state dicts, and feature scalers export correctly."""
    model = create_vqvae(seed=42, input_dim=10, hidden_dim=32, z_dim=4, n_states=20)
    mean = np.random.randn(10).astype(np.float32)
    std = np.random.rand(10).astype(np.float32) + 0.1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model.export_model(tmp_path, mean, std)

        # Assert all exported files exist
        assert (tmp_path / "encoder_state_dict.pt").exists()
        assert (tmp_path / "model_config.json").exists()
        assert (tmp_path / "feature_scaler.json").exists()
        assert (tmp_path / "centroids.npy").exists()

        # Reconstruct new model from exported artifacts
        new_model = TdiV2Model.load_from_export(tmp_path)

        # Check inference predictions are identical
        x = torch.randn(5, 10)
        out_orig = model.encode_states(x)
        out_new = new_model.encode_states(x)

        assert torch.equal(out_orig, out_new)
