"""Plain (no-Lightning) training loop for the single-path v2 alphabet model.

One quantizer per run, fixed LR by default (optional cosine), grad-clip + early-stop on
``val_loss``. Writes the self-describing export plus ``run_config.resolved.json`` and
``train_log.csv`` into the run directory.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tdi.v2.model import AlphabetModel
from tdi.v2.train_config import TrainConfig, _validate_train_config, load_train_config
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
    """Read (virtual_center, max_ca_dist) from the processed-dataset metadata."""
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        preprocessing = manifest.get("preprocessing", {})
        if isinstance(preprocessing, dict):
            return preprocessing.get("virtual_center"), preprocessing.get("max_ca_dist")

    # Backward compatibility with older single-split reports that kept provenance at root.
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
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(y_hat, y)
    raise ValueError(f"Unknown reconstruction loss: {loss_name!r}")


def _run_validation(
    model: AlphabetModel, loader: DataLoader, loss_name: str, n_states: int
) -> dict[str, float]:
    """Compute validation loss components and state diagnostics."""
    model.eval()
    total_recon_loss = 0.0
    total_q_loss = 0.0
    n_examples = 0
    total_margin = 0.0
    margin_examples = 0
    counts = torch.zeros(n_states, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            batch_size = len(x)
            total_recon_loss += float(_reconstruction_loss(loss_name, out["y_hat"], y)) * batch_size
            total_q_loss += float(out["q_loss"]) * batch_size
            n_examples += batch_size
            metrics = out["metrics"]
            if "margin" in metrics:
                total_margin += float(metrics["margin"]) * batch_size
                margin_examples += batch_size
            counts += torch.bincount(out["indices"].cpu(), minlength=n_states)

    if n_examples == 0:
        raise ValueError("Validation loader produced zero examples.")
    val_loss = total_recon_loss / n_examples
    val_q_loss = total_q_loss / n_examples
    dead_state_count = int((counts == 0).sum())
    usage = counts.float() / n_examples
    perplexity = float(torch.exp(-(usage * (usage + 1e-10).log()).sum()))
    diag = {
        # Keep val_loss as reconstruction loss for backward compatibility and early stopping.
        "val_loss": val_loss,
        "val_q_loss": val_q_loss,
        "val_total_loss": val_loss + val_q_loss,
        "perplexity": perplexity,
        "dead_states": dead_state_count,
    }
    if margin_examples:
        diag["margin"] = total_margin / margin_examples
    return diag


def train_model(cfg: TrainConfig) -> AlphabetModel:
    """Run the full training loop and write the export + logs.

    Args:
        cfg: The resolved training configuration.

    Returns:
        The best (lowest val_loss) model, reloaded and exported.
    """
    _validate_train_config(cfg)
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
    if len(val_dataset) == 0:
        raise ValueError("Validation set is empty; provide a non-empty validation split.")

    input_dim = x_train_raw.shape[1]
    if cfg.model.input_dim != input_dim:
        raise ValueError(
            f"model.input_dim={cfg.model.input_dim} does not match training data width {input_dim}."
        )
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
        rotation_trick=cfg.model.rotation_trick,
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

    # Print training headers
    print("\nTraining 3Di VAE v2 model:")
    print(f"  * Quantizer: {cfg.model.quantizer} (n_states={model.n_states})")
    print(f"  * Dataset: {processed_dir.name} (batch_size={cfg.train.batch_size})")
    print(f"  * Output: {out_dir}\n")

    print(
        "Epoch   Train Recon   Train Q   Val Recon   Val Q     "
        "Perplexity   Dead States   Patience   Status"
    )
    print(
        "------------------------------------------------------------------------------------------------"
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.train.patience
    log_rows: list[dict[str, float]] = []

    for epoch in range(cfg.train.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_q_loss = 0.0
        n_batches = 0
        total_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            recon_loss = _reconstruction_loss(cfg.model.loss, out["y_hat"], y)
            q_loss = out["q_loss"]
            loss = recon_loss + q_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_recon_loss += float(recon_loss.detach())
            epoch_q_loss += float(q_loss.detach())
            n_batches += 1

            # Live batch progress bar update (no colors)
            progress = int(100 * (batch_idx + 1) / total_batches)
            bar_len = 20
            filled = int(bar_len * (batch_idx + 1) // total_batches)
            arrow = ">" if filled < bar_len else ""
            dots = "." * (bar_len - filled - (1 if filled < bar_len else 0))
            bar = "=" * filled + arrow + dots
            msg = (
                f"\rEpoch {epoch + 1:2d}/{cfg.train.max_epochs:2d} "
                f"[{bar}] {progress:3d}% | Recon: {recon_loss.item():.4f} "
                f"| Q: {q_loss.item():.4f}"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        if scheduler is not None:
            scheduler.step()

        train_loss = epoch_loss / max(1, n_batches)
        train_recon_loss = epoch_recon_loss / max(1, n_batches)
        train_q_loss = epoch_q_loss / max(1, n_batches)
        diag = _run_validation(model, val_loader, cfg.model.loss, model.n_states)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon_loss": train_recon_loss,
                "train_q_loss": train_q_loss,
                **diag,
            }
        )

        # Clear progress line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        status = ""
        if diag["val_loss"] < best_val:
            best_val = diag["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.train.patience
            status = "New Best"
        else:
            patience_left -= 1
            if diag["dead_states"] > 0:
                status = "Dead states"

        patience_str = f"{patience_left}/{cfg.train.patience}"
        print(
            f" {epoch + 1:<6d} {train_recon_loss:<13.4f} {train_q_loss:<9.4f} "
            f"{diag['val_loss']:<11.4f} {diag['val_q_loss']:<9.4f} "
            f"{diag['perplexity']:<12.2f} {diag['dead_states']:<13d} "
            f"{patience_str:<10s} {status}"
        )

        if patience_left <= 0:
            print(f"\nEarly stopping at epoch {epoch + 1} (no val_loss improvement).")
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
        if not arg.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {arg!r}")
        if "." not in arg[2:]:
            raise ValueError(f"Unknown argument: {arg!r}")
        if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
            raise ValueError(f"Override {arg!r} requires a value")
        try:
            value: object = yaml.safe_load(unknown[i + 1])
        except yaml.YAMLError:
            value = unknown[i + 1]
        overrides[arg[2:]] = value
        i += 2
    return overrides


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: ``python -m tdi.v2 train --config ... [--section.key value ...]``."""
    parser = argparse.ArgumentParser(description="Train the single-path v2 alphabet model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--quantizer", type=str, choices=["vq", "fsq"], help="Convenience for model.quantizer."
    )
    parser.add_argument(
        "--rotation-trick", action="store_true", help="Use rotation-trick gradients for VQ."
    )
    parser.add_argument("--out", type=str, help="Convenience for outputs.out_dir.")
    args, unknown = parser.parse_known_args(argv)

    try:
        overrides = _parse_overrides(unknown)
    except ValueError as exc:
        parser.error(str(exc))
    if args.quantizer is not None:
        overrides["model.quantizer"] = args.quantizer
    if args.rotation_trick:
        overrides["model.rotation_trick"] = True
    if args.out is not None:
        overrides["outputs.out_dir"] = args.out

    cfg = load_train_config(args.config, overrides)
    train_model(cfg)


if __name__ == "__main__":
    main()
