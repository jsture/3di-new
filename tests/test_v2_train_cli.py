import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from tdi.v2.train import main
from tdi.v2.train_config import load_train_config


def test_train_config_loading_and_overrides(tmp_path: Path) -> None:
    """Test loading configuration from YAML with overrides via dotted paths."""
    config_data = {
        "model": {
            "input_dim": 10,
            "hidden_dim": 64,
            "z_dim": 4,
            "n_states": 20,
            "quantizer_type": "ema_vq",
            "loss_type": "smooth_l1",
        },
        "quantizer": {
            "l2_normalize": True,
            "decay": 0.99,
            "commitment_cost": 0.25,
            "min_count": 1.0,
            "replacement_warmup_steps": 500,
            "gradient_mode": "rotation_trick",
            "kmeans_init": True,
        },
        "loss": {
            "lambda_self": 0.05,
            "lambda_usage": 0.001,
            "lambda_contrast": 0.02,
            "temperature": 0.1,
        },
        "training": {
            "batch_size": 512,
            "max_epochs": 20,
            "quantizer_warmup_epochs": 1,
            "aux_ramp_epochs": 1,
            "precision": "32-true",
            "seed": 1,
        },
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "warmup_ratio": 0.03,
            "gradient_clip_val": 1.0,
        },
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "descriptor_jitter_std": 0.0,
            "sampler": "alignment_balanced",
            "alignments_per_batch": 64,
        },
        "outputs": {
            "out_dir": str(tmp_path / "out"),
        },
    }

    config_file = tmp_path / "train_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    # 1. Test config loading directly
    cfg = load_train_config(config_file, {"model.hidden_dim": 128, "training.max_epochs": 5})
    assert cfg.model.hidden_dim == 128
    assert cfg.training.max_epochs == 5
    assert cfg.model.z_dim == 4

    # 2. Test running training CLI main with mock trainer/model
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save mock preprocessed arrays in raw layout format
    dummy_x = np.random.randn(10, 10).astype(np.float32)
    dummy_y = np.random.randn(10, 10).astype(np.float32)
    np.save(processed_dir / "train_x_raw.npy", dummy_x)
    np.save(processed_dir / "train_y_raw.npy", dummy_y)
    np.save(processed_dir / "val_x_raw.npy", dummy_x)
    np.save(processed_dir / "val_y_raw.npy", dummy_y)

    # Save a mock scaler
    np.savez(processed_dir / "scaler.npz", mean=np.zeros(10), std=np.ones(10))

    with (
        patch("tdi.v2.train.TdiV2Model") as mock_model_cls,
        patch("tdi.v2.train.L.Trainer") as mock_trainer_cls,
    ):
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        # Mock sys.argv to pass config and dotted overrides
        test_argv = [
            "train.py",
            "--config",
            str(config_file),
            "--model.hidden_dim",
            "128",
            "--training.max_epochs",
            "5",
            "--quantizer.gradient_mode",
            "ste",
        ]

        with patch.object(sys, "argv", test_argv):
            main()

        # Ensure the model was instantiated with CLI overrides
        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args
        assert kwargs["hidden_dim"] == 128
        assert kwargs["gradient_mode"] == "ste"
        assert kwargs["quantizer_warmup_epochs"] == 1

        # Ensure trainer was created with max_epochs override
        mock_trainer_cls.assert_called_once()
        _, trainer_kwargs = mock_trainer_cls.call_args
        assert trainer_kwargs["max_epochs"] == 5
