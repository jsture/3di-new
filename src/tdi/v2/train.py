"""Plain (no-Lightning) training loop for the single-path v2 alphabet model.

One quantizer per run, fixed LR by default (optional cosine), grad-clip + early-stop on
``val_loss``. Writes the self-describing export plus ``run_config.resolved.json`` and
``train_log.csv`` into the run directory.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tdi.v2.model import AlphabetModel
from tdi.v2.train_config import TrainConfig, load_train_config
from tdi.v2.training_data import PairDataset


def _load_arrays(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val raw descriptor arrays, supporting both layouts.

    Prefers the explicit ``{train,val}_{x,y}_raw.npy`` layout; falls back to a stacked
    ``data.npy`` (train) plus a distinct ``val/data.npy`` (never aliasing the train set).
    """
    train_x_path = processed_dir / "train_x_raw.npy"
    if train_x_path.exists():
        return (
            np.load(processed_dir / "train_x_raw.npy"),
            np.load(processed_dir / "train_y_raw.npy"),
            np.load(processed_dir / "val_x_raw.npy"),
            np.load(processed_dir / "val_y_raw.npy"),
        )

    train_data_path = processed_dir / "data.npy"
    if not train_data_path.exists():
        raise FileNotFoundError(f"No training data files found in {processed_dir}")
    train_data = np.load(train_data_path)

    val_data_path = processed_dir / "val" / "data.npy"
    if not val_data_path.exists():
        raise FileNotFoundError(
            f"Training data found at {train_data_path} but no validation data at "
            f"{val_data_path}. Provide a separate validation split rather than reusing train."
        )
    val_data = np.load(val_data_path)
    return train_data[:, :, 0], train_data[:, :, 1], val_data[:, :, 0], val_data[:, :, 1]


def _read_provenance(processed_dir: Path) -> tuple[list[float] | None, float | None]:
    """Read (virtual_center, max_ca_dist) from a data report if present."""
    for name in ("report.json", "training_data_report.json"):
        path = processed_dir / name
        if path.exists():
            with open(path) as f:
                report = json.load(f)
            return report.get("virtual_center"), report.get("max_ca_dist")
    return None, None


def _reconstruction_loss(loss_name: str, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Partner-prediction reconstruction loss."""
    if loss_name == "mse":
        return F.mse_loss(y_hat, y)
    return F.smooth_l1_loss(y_hat, y)


def _run_validation(
    model: AlphabetModel, loader: DataLoader, loss_name: str, n_states: int
) -> dict[str, float]:
    """Compute val_loss plus state diagnostics over the whole validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    perplexities: list[float] = []
    margins: list[float] = []
    counts = torch.zeros(n_states, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += float(_reconstruction_loss(loss_name, out["y_hat"], y))
            n_batches += 1
            metrics = out["metrics"]
            perplexities.append(float(metrics["perplexity"]))
            if "margin" in metrics:
                margins.append(float(metrics["margin"]))
            counts += torch.bincount(out["indices"].cpu(), minlength=n_states)

    val_loss = total_loss / max(1, n_batches)
    dead_state_count = int((counts == 0).sum())
    diag = {
        "val_loss": val_loss,
        "perplexity": float(np.mean(perplexities)) if perplexities else 0.0,
        "dead_states": dead_state_count,
    }
    if margins:
        diag["margin"] = float(np.mean(margins))
    return diag


def train_model(cfg: TrainConfig) -> AlphabetModel:
    """Run the full training loop and write the export + logs.

    Args:
        cfg: The resolved training configuration.

    Returns:
        The best (lowest val_loss) model, reloaded and exported.
    """
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    processed_dir = Path(cfg.data.processed_dir)
    x_train_raw, y_train_raw, x_val_raw, y_val_raw = _load_arrays(processed_dir)

    # Train-only scaler: fit on train, reuse for val (no leakage).
    scaler_path = processed_dir / "scaler.npz"
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        mean, std = scaler["mean"], scaler["std"]
        train_dataset = PairDataset(x_train_raw, y_train_raw, mean=mean, std=std)
    else:
        train_dataset = PairDataset(x_train_raw, y_train_raw, fit_scaler=True)
    mean, std = train_dataset.mean, train_dataset.std
    val_dataset = PairDataset(x_val_raw, y_val_raw, mean=mean, std=std)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # drop_last=True yields zero batches (and a barely-initialized export) when the training
    # set is smaller than one batch; fail loudly instead.
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set has {len(train_dataset)} examples and batch_size="
            f"{cfg.train.batch_size} with drop_last=True, producing zero batches. "
            "Lower batch_size or provide more training data."
        )

    input_dim = x_train_raw.shape[1]
    model = AlphabetModel(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        z_dim=cfg.model.z_dim,
        n_states=cfg.model.n_states,
        quantizer=cfg.model.quantizer,
        levels=cfg.model.levels,
        loss=cfg.model.loss,
        decay=cfg.model.decay,
        commitment_cost=cfg.model.commitment_cost,
        min_count=cfg.model.min_count,
        l2_normalize=cfg.model.l2_normalize,
        replacement_warmup_steps=cfg.model.replacement_warmup_steps,
    )

    # One-shot k-means codebook init on the VQ path (no-op for FSQ).
    if cfg.model.quantizer in ("vq", "ema_vq") and cfg.train.kmeans_init:
        model.init_codebook_from_loader(
            train_loader,
            n_batches=cfg.train.kmeans_init_batches,
            seed=cfg.train.kmeans_seed,
        )

    # AdamW with no weight decay on biases / LayerNorm gains.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)
        if cfg.train.scheduler == "cosine"
        else None
    )

    out_dir = Path(cfg.outputs.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.train.patience
    log_rows: list[dict[str, float]] = []

    for epoch in range(cfg.train.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = _reconstruction_loss(cfg.model.loss, out["y_hat"], y) + out["q_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()

        train_loss = epoch_loss / max(1, n_batches)
        diag = _run_validation(model, val_loader, cfg.model.loss, model.n_states)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, **diag})
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={diag['val_loss']:.4f} "
            f"perplexity={diag['perplexity']:.2f} dead_states={diag['dead_states']}"
        )

        if diag["val_loss"] < best_val:
            best_val = diag["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.train.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement).")
                break

    # Restore the best weights before exporting.
    if best_state is not None:
        model.load_state_dict(best_state)

    virtual_center, max_ca_dist = _read_provenance(processed_dir)
    model.save(out_dir, mean=mean, std=std, virtual_center=virtual_center, max_ca_dist=max_ca_dist)

    with open(out_dir / "run_config.resolved.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    if log_rows:
        fieldnames = list(log_rows[-1].keys())
        with open(out_dir / "train_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in log_rows:
                writer.writerow(row)

    print(f"Exported model artifacts and logs to {out_dir}")
    return model


def _parse_overrides(unknown: list[str]) -> dict[str, object]:
    """Parse ``--section.key value`` overrides, value-typed via YAML."""
    import yaml

    overrides: dict[str, object] = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--") and i + 1 < len(unknown):
            try:
                value: object = yaml.safe_load(unknown[i + 1])
            except yaml.YAMLError:
                value = unknown[i + 1]
            overrides[arg[2:]] = value
            i += 2
        elif arg.startswith("--"):
            overrides[arg[2:]] = True
            i += 1
        else:
            i += 1
    return overrides


def main() -> None:
    """CLI entrypoint: ``python -m tdi.v2 train --config ... [--section.key value ...]``."""
    parser = argparse.ArgumentParser(description="Train the single-path v2 alphabet model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--quantizer", type=str, choices=["vq", "fsq"], help="Convenience for model.quantizer."
    )
    parser.add_argument("--out", type=str, help="Convenience for outputs.out_dir.")
    args, unknown = parser.parse_known_args()

    overrides = _parse_overrides(unknown)
    if args.quantizer is not None:
        overrides["model.quantizer"] = args.quantizer
    if args.out is not None:
        overrides["outputs.out_dir"] = args.out

    cfg = load_train_config(args.config, overrides)
    train_model(cfg)


if __name__ == "__main__":
    main()
