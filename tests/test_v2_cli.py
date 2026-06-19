# ruff: noqa: E402, I001
"""Unit tests for modernized train_v2.py config loading and CLI overrides."""

import sys
from pathlib import Path

# Add project root to sys.path to make top-level directories like scripts importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
from unittest.mock import patch, MagicMock
import numpy as np

from scripts.train_v2 import main


def test_train_v2_config_and_overrides(tmp_path) -> None:
    """Test loading configuration from YAML with CLI overrides."""
    # 1. Create a dummy YAML config file
    config_data = {
        "model": {
            "input_dim": 10,
            "hidden_dim": 64,
            "z_dim": 4,
            "n_states": 20,
            "quantizer": "vq",
            "kmeans_init": False,
        },
        "loss": {
            "commitment_weight": 0.25,
            "usage_weight": 0.0,
            "contrastive_weight": 0.0,
            "self_reconstruction_weight": 0.1,
        },
        "data": {
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
            "descriptor_jitter_std": 0.0,
            "alignments_per_batch": 0,
        },
        "optimizer": {
            "lr": 0.005,
        },
        "training": {
            "seed": 123,
            "out_dir": str(tmp_path / "out"),
            "max_epochs": 10,
            "batch_size": 256,
            "continuous_warmup_epochs": 0,
            "aux_ramp_epochs": 0,
            "accumulate_grad_batches": 1,
            "precision": "32-true",
            "torch_compile": False,
        },
    }

    config_file = tmp_path / "train_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    # 2. Mock dataset arrays so np.load returns valid mock arrays
    dummy_data = np.random.randn(20, 10, 2).astype(np.float32)

    (tmp_path / "train").mkdir(exist_ok=True, parents=True)
    (tmp_path / "val").mkdir(exist_ok=True, parents=True)
    np.save(tmp_path / "train" / "data.npy", dummy_data)
    np.save(tmp_path / "val" / "data.npy", dummy_data)

    # 3. Patch TdiV2Model, L.Trainer, and torch.compile
    with (
        patch("scripts.train_v2.TdiV2Model") as mock_model_cls,
        patch("scripts.train_v2.L.Trainer") as mock_trainer_cls,
        patch("scripts.train_v2.torch.compile") as mock_compile,
    ):
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        # Mock sys.argv
        test_argv = [
            "train_v2.py",
            "--config",
            str(config_file),
            "--kmeans-init",
            "--continuous-warmup-epochs",
            "2",
            "--contrastive-weight",
            "0.05",
            "--precision",
            "bf16-mixed",
            "--torch-compile",
        ]

        with patch.object(sys, "argv", test_argv):
            main()

        # 4. Asserts
        # Ensure the model was instantiated with the correct overridden parameters
        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args

        assert kwargs["kmeans_init"] is True
        assert kwargs["quantizer_warmup_epochs"] == 2
        assert kwargs["lambda_contrast"] == 0.05
        assert kwargs["n_states"] == 20
        assert kwargs["lr"] == 0.005

        # Ensure model compilation was called because `--torch-compile` was set
        mock_compile.assert_called_once_with(mock_model)

        # Ensure trainer was created with overridden precision
        mock_trainer_cls.assert_called_once()
        _, trainer_kwargs = mock_trainer_cls.call_args
        assert trainer_kwargs["precision"] == "bf16-mixed"


def test_train_v2_config_only(tmp_path) -> None:
    """Test loading configuration from YAML with no overrides."""
    # Create YAML config
    config_data = {
        "model": {
            "n_states": 20,
            "quantizer": "vq",
            "kmeans_init": False,
        },
        "data": {
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
        },
        "optimizer": {
            "lr": 0.001,
        },
        "training": {
            "seed": 42,
            "out_dir": str(tmp_path / "out"),
            "max_epochs": 100,
            "batch_size": 512,
        },
    }
    config_file = tmp_path / "train_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    dummy_data = np.random.randn(20, 10, 2).astype(np.float32)
    (tmp_path / "train").mkdir(exist_ok=True, parents=True)
    (tmp_path / "val").mkdir(exist_ok=True, parents=True)
    np.save(tmp_path / "train" / "data.npy", dummy_data)
    np.save(tmp_path / "val" / "data.npy", dummy_data)

    with (
        patch("scripts.train_v2.TdiV2Model") as mock_model_cls,
        patch("scripts.train_v2.L.Trainer") as mock_trainer_cls,
        patch("scripts.train_v2.torch.compile") as mock_compile,
    ):
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        test_argv = [
            "train_v2.py",
            "--config",
            str(config_file),
        ]

        with patch.object(sys, "argv", test_argv):
            main()

        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args

        # Verify defaults from yaml config
        assert kwargs["kmeans_init"] is False
        assert kwargs["quantizer_warmup_epochs"] == 0  # model default
        assert kwargs["lambda_contrast"] == 0.0  # model default
        assert kwargs["n_states"] == 20
        assert kwargs["lr"] == 0.001

        # torch.compile should not be called since torch_compile is false in defaults
        mock_compile.assert_not_called()
