"""Tests for the slim v2 training config loader and CLI override plumbing."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tdi.v2.train import main
from tdi.v2.train_config import TrainConfig, load_train_config


def _write_config(tmp_path: Path) -> Path:
    config = {
        "model": {"quantizer": "vq", "hidden_dim": 64, "z_dim": 4, "n_states": 20},
        "train": {"lr": 0.005, "max_epochs": 20, "batch_size": 256, "kmeans_init": False},
        "data": {"processed_dir": str(tmp_path / "processed")},
        "outputs": {"out_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "train_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_load_config_applies_dotted_overrides(tmp_path: Path) -> None:
    """Dotted overrides patch nested sections and types parse via YAML."""
    config_file = _write_config(tmp_path)
    cfg = load_train_config(
        config_file, {"model.hidden_dim": 128, "train.max_epochs": 5, "model.levels": [5, 5]}
    )
    assert cfg.model.hidden_dim == 128
    assert cfg.train.max_epochs == 5
    assert cfg.model.levels == [5, 5]
    assert cfg.model.z_dim == 4  # untouched


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model.loss": "smoothl1"}, "model.loss"),
        ({"train.scheduler": "cosin"}, "train.scheduler"),
        ({"trian.lr": 0.1}, "Unknown training config section"),
    ],
)
def test_load_config_rejects_fail_open_values(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    """Typos cannot silently select fallback training behavior."""
    with pytest.raises(ValueError, match=message):
        load_train_config(_write_config(tmp_path), override)


def test_main_resolves_cli_overrides_and_conveniences(tmp_path: Path) -> None:
    """main() merges --config, dotted overrides, and --quantizer/--out conveniences."""
    config_file = _write_config(tmp_path)

    captured: dict[str, TrainConfig] = {}

    def _capture(cfg: TrainConfig) -> None:
        captured["cfg"] = cfg

    test_argv = [
        "train",
        "--config",
        str(config_file),
        "--train.lr",
        "0.01",
        "--model.n_states",
        "24",
        "--quantizer",
        "fsq",
        "--out",
        str(tmp_path / "run_x"),
    ]
    with (
        patch("tdi.v2.train.train_model", side_effect=_capture),
        patch.object(sys, "argv", test_argv),
    ):
        main()

    cfg = captured["cfg"]
    assert cfg.model.quantizer == "fsq"
    assert cfg.model.n_states == 24
    assert cfg.train.lr == 0.01
    assert cfg.outputs.out_dir == str(tmp_path / "run_x")


def test_main_enables_rotation_trick_flag(tmp_path: Path) -> None:
    """The convenience flag enables rotation-trick gradients in the resolved config."""
    captured: dict[str, TrainConfig] = {}

    with patch("tdi.v2.train.train_model", side_effect=lambda cfg: captured.setdefault("cfg", cfg)):
        main(["--config", str(_write_config(tmp_path)), "--rotation-trick"])

    assert captured["cfg"].model.rotation_trick is True
