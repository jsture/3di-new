import argparse
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from tdi.v2.model import TdiV2Model
from tdi.v2.train_config import TrainConfig, load_train_config
from tdi.v2.training_data import AlignmentBatchSampler, PairDataset


class _EpochSeedingCallback(Callback):
    """Propagate the epoch into the dataset and batch sampler each train epoch.

    Without this, ``PairDataset``/``AlignmentBatchSampler`` keep ``epoch == 0`` for the
    whole run, so per-epoch descriptor jitter and batch composition never vary.
    """

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch
        loader = trainer.train_dataloader
        if loader is None:
            return
        for obj in (getattr(loader, "dataset", None), getattr(loader, "batch_sampler", None)):
            set_epoch = getattr(obj, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)


def _load_alignment_ids(data_dir: str | Path, n_rows: int) -> np.ndarray | None:
    """Load per-row alignment ids from a metadata parquet if present and row-aligned.

    Args:
        data_dir: Directory containing processed data files.
        n_rows: Number of rows expected in metadata.

    Returns:
        Array of alignment IDs or None if metadata is not found or mismatch.
    """
    import pandas as pd

    data_path = Path(data_dir)
    for name in ("train_metadata.parquet", "metadata.parquet"):
        path = data_path / name
        if path.exists():
            df = pd.read_parquet(path)
            if "alignment_id" in df.columns and len(df) == n_rows:
                return df["alignment_id"].to_numpy()
    return None


def train_model(cfg: TrainConfig) -> None:
    """Execute the full model training loop using the provided TrainConfig.

    Args:
        cfg: The training configuration object.
    """
    # Seed all sources of randomness
    L.seed_everything(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    processed_dir = Path(cfg.data.processed_dir)

    # Load preprocessed arrays. Try new raw name layout first, then fallback to old data.
    train_x_path = processed_dir / "train_x_raw.npy"
    if train_x_path.exists():
        x_train_raw = np.load(processed_dir / "train_x_raw.npy")
        y_train_raw = np.load(processed_dir / "train_y_raw.npy")
        x_val_raw = np.load(processed_dir / "val_x_raw.npy")
        y_val_raw = np.load(processed_dir / "val_y_raw.npy")
    else:
        # Fallback to single stacked data.npy if raw layout is not present
        train_data_path = processed_dir / "data.npy"
        if train_data_path.exists():
            train_data_raw = np.load(train_data_path)
            x_train_raw = train_data_raw[:, :, 0]
            y_train_raw = train_data_raw[:, :, 1]

            # Validation data must be a distinct file; never silently alias the train set.
            val_data_path = processed_dir / "val" / "data.npy"
            if not val_data_path.exists():
                raise FileNotFoundError(
                    f"Training data found at {train_data_path} but no validation data at "
                    f"{val_data_path}. Provide a separate validation split rather than "
                    "reusing the training data."
                )
            val_data_raw = np.load(val_data_path)
            x_val_raw = val_data_raw[:, :, 0]
            y_val_raw = val_data_raw[:, :, 1]
        else:
            raise FileNotFoundError(f"No training data files found in {processed_dir}")

    # Load scaler statistics if saved during preprocessing
    scaler_path = processed_dir / "scaler.npz"
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        mean = scaler["mean"]
        std = scaler["std"]
        print(f"Loaded standardizer scaler from {scaler_path}")
    else:
        mean = None
        std = None
        print(
            "No scaler.npz found in processed data directory; "
            "fitting standardizer from training data."
        )

    # Create PairDataset for train split (supports descriptor jittering)
    train_dataset = PairDataset(
        x_train_raw,
        y_train_raw,
        mean=mean,
        std=std,
        descriptor_jitter_std=cfg.data.descriptor_jitter_std,
        seed=cfg.training.seed,
    )
    # Obtain standardized metrics (fitted or loaded)
    mean = train_dataset.mean
    std = train_dataset.std

    # Create PairDataset for validation split (no jitter)
    val_dataset = PairDataset(
        x_val_raw,
        y_val_raw,
        mean=mean,
        std=std,
        descriptor_jitter_std=0.0,
        seed=cfg.training.seed,
    )

    # Resolve loader batch sampler
    sampler = None
    if cfg.data.sampler == "alignment_balanced" and cfg.data.alignments_per_batch:
        alignment_ids = _load_alignment_ids(processed_dir, len(x_train_raw))
        if alignment_ids is not None:
            sampler = AlignmentBatchSampler(
                alignment_ids,
                batch_size=cfg.training.batch_size,
                alignments_per_batch=cfg.data.alignments_per_batch,
                seed=cfg.training.seed,
            )
            print("Using AlignmentBatchSampler for training.")
        else:
            print(
                "Alignment-aware batching requested but alignment metadata "
                "not found/mismatched; using random sampler."
            )

    # Determine if workers should persist (only valid when num_workers > 0)
    persistent = cfg.data.num_workers > 0

    if sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.data.num_workers,
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            persistent_workers=persistent,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=persistent,
    )

    # Initialize TdiV2Model
    input_dim = x_train_raw.shape[1]

    # Map model quantizer settings to model properties
    quantizer_param = cfg.model.quantizer_type
    quantizer_type = "vq" if quantizer_param in ("vq", "ema_vq") else quantizer_param

    model = TdiV2Model(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        z_dim=cfg.model.z_dim,
        n_states=cfg.model.n_states,
        quantizer_type=quantizer_type,
        fsq_levels=cfg.model.fsq_levels,
        decay=cfg.quantizer.decay,
        eps=1e-5,  # smooth epsilon
        commitment_cost=cfg.quantizer.commitment_cost,
        l2_normalize=cfg.quantizer.l2_normalize,
        min_count=cfg.quantizer.min_count,
        replacement_warmup_steps=cfg.quantizer.replacement_warmup_steps,
        lambda_usage=cfg.loss.lambda_usage,
        lambda_contrast=cfg.loss.lambda_contrast,
        lambda_self=cfg.loss.lambda_self,
        temperature=cfg.loss.temperature,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_ratio=cfg.optimizer.warmup_ratio,
        quantizer_warmup_epochs=cfg.training.quantizer_warmup_epochs,
        aux_ramp_epochs=cfg.training.aux_ramp_epochs,
        loss_type=cfg.model.loss_type,
        kmeans_init=cfg.quantizer.kmeans_init,
        kmeans_init_batches=cfg.quantizer.kmeans_init_batches,
        kmeans_seed=cfg.quantizer.kmeans_seed,
        kmeans_max_samples=cfg.quantizer.kmeans_init_samples,
        gradient_mode=cfg.quantizer.gradient_mode,
    )

    out_dir = Path(cfg.outputs.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup saving callbacks and EarlyStopping
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_score:.2f}",
        dirpath=str(out_dir),
    )
    early_stopping = EarlyStopping(
        monitor="val_score",
        mode="max",
        patience=10,
    )

    logger = CSVLogger(
        save_dir=str(out_dir),
        name="logs",
    )

    # Configure Lightning Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, _EpochSeedingCallback()],
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm="norm",
        default_root_dir=str(out_dir),
    )

    # Run fitting
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load and save the best checkpoint
    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"Loading best checkpoint from {best_path}")
        best_model = TdiV2Model.load_from_checkpoint(best_path)
    else:
        print("No checkpoint saved, exporting current model.")
        best_model = model

    # Read feature-build provenance (virtual center / Ca filter) from the data report if
    # present, so the exported config records what was actually used rather than guessing.
    virtual_center = None
    max_ca_dist = None
    report_path = processed_dir / "training_data_report.json"
    if report_path.exists():
        with open(report_path) as f:
            data_report = json.load(f)
        virtual_center = data_report.get("virtual_center")
        max_ca_dist = data_report.get("max_ca_dist")

    # Save best model to export files
    best_model.export_model(
        out_dir,
        mean=mean,
        std=std,
        virtual_center=virtual_center,
        max_ca_dist=max_ca_dist,
    )

    # Save training_config.yaml in the output directory
    with open(out_dir / "training_config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, default_flow_style=False)

    print(f"Exported model artifacts and config to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train modernized VQ-VAE (v2) model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training configuration file.",
    )
    args, unknown = parser.parse_known_args()

    # Parse dotted section.key overrides from unknown arguments
    overrides = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            dotted = arg[2:]
            if i + 1 < len(unknown):
                val_str = unknown[i + 1]
                if val_str.lower() == "true":
                    val = True
                elif val_str.lower() == "false":
                    val = False
                elif val_str.lower() in ("null", "none"):
                    val = None
                else:
                    try:
                        if "." in val_str:
                            val = float(val_str)
                        else:
                            val = int(val_str)
                    except ValueError:
                        val = val_str
                overrides[dotted] = val
                i += 2
            else:
                overrides[dotted] = True
                i += 1
        else:
            i += 1

    # Load configuration file
    cfg = load_train_config(args.config, overrides)
    train_model(cfg)


if __name__ == "__main__":
    main()
