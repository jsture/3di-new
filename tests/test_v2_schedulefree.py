"""Tests for the Schedule-Free AdamW optimizer option.

Schedule-Free keeps two parameter views -- the gradient-evaluation point ``y`` used while
stepping and the averaged point ``x`` that the method actually returns -- and swaps the live
parameters between them on ``optimizer.train()`` / ``optimizer.eval()``. Reading weights in
the wrong mode exports the wrong iterate silently, so that contract is what these tests pin.
"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from schedulefree import AdamWScheduleFree

from tdi.v2.model import AlphabetModel
from tdi.v2.train import _build_optimizer, _set_optimizer_mode, main, train_model
from tdi.v2.train_config import (
    DataConfig,
    LoopConfig,
    ModelConfig,
    OutputsConfig,
    TrainConfig,
    load_train_config,
)


def _config(tmp_path: Path, optimizer: str, **loop: object) -> TrainConfig:
    """Build a tiny training config; only the end-to-end test needs the processed dir."""
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        np.save(processed / f"{split}_x_raw.npy", rng.standard_normal((64, 10)).astype(np.float32))
        np.save(processed / f"{split}_y_raw.npy", rng.standard_normal((64, 10)).astype(np.float32))
    np.savez(
        processed / "scaler.npz",
        mean=np.zeros(10, dtype=np.float32),
        std=np.ones(10, dtype=np.float32),
    )
    return TrainConfig(
        model=ModelConfig(quantizer="vq", n_states=8, z_dim=4),
        train=replace(
            LoopConfig(batch_size=16, max_epochs=1, kmeans_init=False, optimizer=optimizer),
            **loop,
        ),
        data=DataConfig(processed_dir=str(processed)),
        outputs=OutputsConfig(out_dir=str(tmp_path / "run")),
    )


def test_schedulefree_rejects_an_lr_schedule(tmp_path: Path) -> None:
    """Stacking a cosine schedule on Schedule-Free contradicts the method, so it is refused."""
    config_file = tmp_path / "train_config.yaml"
    config_file.write_text("train:\n  lr: 0.001\n")
    with pytest.raises(ValueError, match="replaces the LR schedule"):
        load_train_config(
            config_file, {"train.optimizer": "schedulefree", "train.scheduler": "cosine"}
        )


def test_build_optimizer_returns_schedulefree_and_no_scheduler(tmp_path: Path) -> None:
    """The Schedule-Free path builds the right optimizer, passes its knobs, takes no schedule."""
    cfg = _config(tmp_path, "schedulefree", lr=0.01, sf_warmup_steps=13, sf_beta=0.98)
    optimizer, scheduler = _build_optimizer(cfg, AlphabetModel(input_dim=10, n_states=8, z_dim=4))

    assert isinstance(optimizer, AdamWScheduleFree)
    assert scheduler is None
    group = optimizer.param_groups[0]
    assert (group["lr"], group["warmup_steps"], group["betas"][0]) == (0.01, 13, 0.98)


def test_build_optimizer_keeps_adamw_and_cosine_intact(tmp_path: Path) -> None:
    """Hoisting optimizer construction into a helper leaves the default path unchanged."""
    cfg = _config(tmp_path, "adamw", scheduler="cosine")
    optimizer, scheduler = _build_optimizer(cfg, AlphabetModel(input_dim=10, n_states=8, z_dim=4))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_set_optimizer_mode_swaps_between_the_y_and_x_iterates() -> None:
    """Train mode exposes y and eval mode exposes the averaged x, so the mode call matters."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = AdamWScheduleFree([param], lr=0.1, warmup_steps=0)

    # Several steps with distinct gradients: at t=1 the averaging weight c_1 is 1, so x, y and
    # z all coincide and the two views are only separable once the average starts to lag.
    optimizer.train()
    for scale in (1.0, -2.0, 0.5):
        param.grad = torch.full((2, 2), scale)
        optimizer.step()

    y_iterate = param.detach().clone()
    _set_optimizer_mode(optimizer, training=False)
    assert not torch.allclose(y_iterate, param.detach())

    # Toggling back restores the gradient-evaluation point exactly.
    _set_optimizer_mode(optimizer, training=True)
    assert torch.allclose(param.detach(), y_iterate)


def test_schedulefree_run_exports_the_eval_iterate(tmp_path: Path) -> None:
    """The exported weights are the averaged x iterate, not the y point used for stepping.

    Regression guard for the checkpointing contract: dropping the eval-mode swap before
    validation and export silently ships y, which is not the iterate the method returns.
    """
    cfg = _config(tmp_path, "schedulefree", lr=0.05, sf_warmup_steps=0)

    captured: list[torch.optim.Optimizer] = []
    real_build = _build_optimizer

    def _spy(config: TrainConfig, model: AlphabetModel) -> object:
        built = real_build(config, model)
        captured.append(built[0])
        return built

    with patch("tdi.v2.train._build_optimizer", side_effect=_spy):
        model = train_model(cfg)

    optimizer = captured[0]
    assert isinstance(optimizer, AdamWScheduleFree)
    assert optimizer.param_groups[0]["train_mode"] is False

    exported = torch.load(tmp_path / "run" / "encoder_state_dict.pt", weights_only=True)
    reference = next(iter(exported))
    assert torch.allclose(exported[reference], model.encoder.state_dict()[reference])

    # Flipping back to the training view must move the weights, proving the export was not
    # taken from y by accident.
    optimizer.train()
    assert not torch.allclose(exported[reference], model.encoder.state_dict()[reference])


def test_main_selects_schedulefree_via_the_convenience_flag(tmp_path: Path) -> None:
    """``--optimizer schedulefree`` resolves into the config like --quantizer does."""
    config_file = tmp_path / "train_config.yaml"
    config_file.write_text("train:\n  lr: 0.001\n")

    captured: dict[str, TrainConfig] = {}
    with patch("tdi.v2.train.train_model", side_effect=lambda cfg: captured.setdefault("cfg", cfg)):
        main(["--config", str(config_file), "--optimizer", "schedulefree"])

    assert captured["cfg"].train.optimizer == "schedulefree"
