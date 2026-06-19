#!/usr/bin/env python3
"""CLI: train modernized VQ-VAE (v2) model to learn discrete 3Di state representations."""

import argparse
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from tdi.v2.model import TdiV2Model
from tdi.v2.training_data import PairDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train modernized VQ-VAE (v2) model using PyTorch Lightning."
    )
    parser.add_argument("seed", type=int, help="Seed value for reproducibility.")
    parser.add_argument("train_dir", type=str, help="Directory containing training data artifacts.")
    parser.add_argument("val_dir", type=str, help="Directory containing validation data artifacts.")
    parser.add_argument(
        "out_dir", type=str, help="Output directory to save trained model parameters."
    )
    parser.add_argument("n_states", type=int, help="Size of discrete 3Di alphabet states.")
    parser.add_argument(
        "--quantizer_type",
        type=str,
        default="vq",
        choices=["vq", "fsq"],
        help="Quantization backend: vq or fsq.",
    )
    parser.add_argument(
        "--fsq_levels",
        type=int,
        nargs="+",
        default=None,
        help="FSQ grid levels (e.g., --fsq_levels 5 4).",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--jitter_std", type=float, default=0.0, help="Coordinate jittering noise std."
    )
    args = parser.parse_args()

    # Seed all sources of randomness
    L.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    # Load pair data
    train_data_raw = np.load(os.path.join(args.train_dir, "data.npy"))
    x_train_raw = train_data_raw[:, :, 0]
    y_train_raw = train_data_raw[:, :, 1]

    val_data_raw = np.load(os.path.join(args.val_dir, "data.npy"))
    x_val_raw = val_data_raw[:, :, 0]
    y_val_raw = val_data_raw[:, :, 1]

    # Create train dataset (fits scaler)
    train_dataset = PairDataset(
        x_train_raw, y_train_raw, jitter_std=args.jitter_std, seed=args.seed
    )
    mean = train_dataset.mean
    std = train_dataset.std

    # Create val dataset (uses train scaler, NO jitter)
    val_dataset = PairDataset(
        x_val_raw, y_val_raw, mean=mean, std=std, jitter_std=0.0, seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    input_dim = x_train_raw.shape[1]
    hidden_dim = 64
    z_dim = 4

    model = TdiV2Model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        n_states=args.n_states,
        quantizer_type=args.quantizer_type,
        fsq_levels=args.fsq_levels,
        lr=args.lr,
    )

    # Callbacks for validation score selection
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_score:.2f}",
        dirpath=args.out_dir,
    )
    early_stopping = EarlyStopping(monitor="val_score", mode="max", patience=10)

    # Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint
    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"Loading best checkpoint from {best_path}")
        best_model = TdiV2Model.load_from_checkpoint(best_path)
    else:
        print("No checkpoint saved, exporting current model.")
        best_model = model

    # Export best model
    best_model.export_model(args.out_dir, mean=mean, std=std)
    print(f"Exported best model and scaler to {args.out_dir}")


if __name__ == "__main__":
    main()
