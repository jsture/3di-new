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
from tdi.v2.training_data import AlignmentBatchSampler, PairDataset


def _load_alignment_ids(data_dir: str, n_rows: int) -> np.ndarray | None:
    """Load per-row alignment ids from a metadata parquet if present and row-aligned.

    Looks for ``train_metadata.parquet`` then ``metadata.parquet`` in ``data_dir``; returns
    the ``alignment_id`` column only when it exists and matches ``n_rows``, else None.
    """
    import pandas as pd

    for name in ("train_metadata.parquet", "metadata.parquet"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "alignment_id" in df.columns and len(df) == n_rows:
                return df["alignment_id"].to_numpy()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train modernized VQ-VAE (v2) model using PyTorch Lightning."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML training configuration file."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed value for reproducibility.")
    parser.add_argument(
        "--train-dir",
        "--train_dir",
        dest="train_dir",
        type=str,
        default=None,
        help="Directory containing training data artifacts.",
    )
    parser.add_argument(
        "--val-dir",
        "--val_dir",
        dest="val_dir",
        type=str,
        default=None,
        help="Directory containing validation data artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=None,
        help="Output directory to save trained model parameters.",
    )
    parser.add_argument(
        "--n-states",
        "--n_states",
        dest="n_states",
        type=int,
        default=None,
        help="Size of discrete 3Di alphabet states.",
    )
    parser.add_argument(
        "--quantizer-type",
        "--quantizer_type",
        dest="quantizer_type",
        type=str,
        choices=["vq", "fsq"],
        default=None,
        help="Quantization backend: vq or fsq.",
    )
    parser.add_argument(
        "--fsq-levels",
        "--fsq_levels",
        dest="fsq_levels",
        type=int,
        nargs="+",
        default=None,
        help="FSQ grid levels (e.g., --fsq_levels 5 4).",
    )
    parser.add_argument(
        "--max-epochs",
        "--max_epochs",
        dest="max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Mini-batch size.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")

    # Quantizer & Regularizers overrides
    parser.add_argument(
        "--kmeans-init",
        dest="kmeans_init",
        action="store_true",
        default=None,
        help="Enable k-means initialization of codebook centroids.",
    )
    parser.add_argument(
        "--no-kmeans-init",
        dest="kmeans_init",
        action="store_false",
        default=None,
        help="Disable k-means initialization of codebook centroids.",
    )
    parser.add_argument(
        "--continuous-warmup-epochs",
        "--quantizer-warmup-epochs",
        "--quantizer_warmup_epochs",
        dest="quantizer_warmup_epochs",
        type=int,
        default=None,
        help="Number of warmup epochs using continuous latents before quantization begins.",
    )
    parser.add_argument(
        "--aux-ramp-epochs",
        dest="aux_ramp_epochs",
        type=int,
        default=None,
        help="Number of epochs to ramp auxiliary loss components from 0 to 1.",
    )
    parser.add_argument(
        "--contrastive-weight",
        "--lambda-contrast",
        "--lambda_contrast",
        dest="lambda_contrast",
        type=float,
        default=None,
        help="Contrastive auxiliary loss weight lambda_contrast.",
    )
    parser.add_argument(
        "--usage-weight",
        "--lambda-usage",
        "--lambda_usage",
        dest="lambda_usage",
        type=float,
        default=None,
        help="Code usage entropy weight lambda_usage.",
    )
    parser.add_argument(
        "--self-weight",
        "--lambda-self",
        "--lambda_self",
        dest="lambda_self",
        type=float,
        default=None,
        help="Self-reconstruction loss weight lambda_self.",
    )
    parser.add_argument(
        "--descriptor-jitter-std",
        "--descriptor_jitter_std",
        dest="descriptor_jitter_std",
        type=float,
        default=None,
        help="Descriptor-space jitter std applied in PairDataset.",
    )

    # Hardware & Batch overrides
    parser.add_argument(
        "--alignments-per-batch",
        "--alignments_per_batch",
        dest="alignments_per_batch",
        type=int,
        default=None,
        help="Number of distinct alignments per batch in alignment-aware sampler.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        "--accumulate_grad_batches",
        dest="accumulate_grad_batches",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--precision", type=str, default=None, help="Trainer precision (e.g. 32-true, bf16-mixed)."
    )
    parser.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        default=None,
        help="Enable model compilation via torch.compile.",
    )
    parser.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        default=None,
        help="Disable model compilation via torch.compile.",
    )

    # Miscellaneous model/training config overrides
    parser.add_argument("--decay", type=float, default=None, help="EMA decay rate.")
    parser.add_argument("--eps", type=float, default=None, help="Laplace smoothing epsilon.")
    parser.add_argument(
        "--commitment-cost",
        "--commitment_cost",
        dest="commitment_cost",
        type=float,
        default=None,
        help="Commitment cost for quantization loss.",
    )
    parser.add_argument(
        "--l2-normalize",
        dest="l2_normalize",
        action="store_true",
        default=None,
        help="Use L2 normalization for distance lookup.",
    )
    parser.add_argument(
        "--no-l2-normalize",
        dest="l2_normalize",
        action="store_false",
        default=None,
        help="Disable L2 normalization for distance lookup.",
    )
    parser.add_argument(
        "--min-count",
        "--min_count",
        dest="min_count",
        type=float,
        default=None,
        help="Minimum count for dead code replacement.",
    )
    parser.add_argument(
        "--replacement-warmup-steps",
        "--replacement_warmup_steps",
        dest="replacement_warmup_steps",
        type=int,
        default=None,
        help="Warmup steps before dead codebook replacements occur.",
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=None,
        help="Weight decay for optimization.",
    )
    parser.add_argument(
        "--warmup-ratio",
        "--warmup_ratio",
        dest="warmup_ratio",
        type=float,
        default=None,
        help="Warmup step ratio for LR schedule.",
    )
    parser.add_argument(
        "--loss-type",
        "--loss_type",
        dest="loss_type",
        type=str,
        default=None,
        help="Reconstruction loss type: smooth_l1 or gaussian_nll.",
    )
    args = parser.parse_args()

    import yaml

    # Load configuration from yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Map CLI options to nested config keys (section, key)
    cli_mapping = {
        "seed": ("training", "seed"),
        "train_dir": ("data", "train_dir"),
        "val_dir": ("data", "val_dir"),
        "out_dir": ("training", "out_dir"),
        "n_states": ("model", "n_states"),
        "quantizer_type": ("model", "quantizer"),
        "fsq_levels": ("model", "fsq_levels"),
        "max_epochs": ("training", "max_epochs"),
        "batch_size": ("training", "batch_size"),
        "lr": ("optimizer", "lr"),
        "kmeans_init": ("model", "kmeans_init"),
        "quantizer_warmup_epochs": ("training", "continuous_warmup_epochs"),
        "aux_ramp_epochs": ("training", "aux_ramp_epochs"),
        "lambda_contrast": ("loss", "contrastive_weight"),
        "lambda_usage": ("loss", "usage_weight"),
        "lambda_self": ("loss", "self_reconstruction_weight"),
        "descriptor_jitter_std": ("data", "descriptor_jitter_std"),
        "alignments_per_batch": ("data", "alignments_per_batch"),
        "accumulate_grad_batches": ("training", "accumulate_grad_batches"),
        "precision": ("training", "precision"),
        "torch_compile": ("training", "torch_compile"),
        "decay": ("model", "decay"),
        "eps": ("model", "eps"),
        "commitment_cost": ("loss", "commitment_weight"),
        "l2_normalize": ("model", "l2_normalize"),
        "min_count": ("model", "min_count"),
        "replacement_warmup_steps": ("model", "replacement_warmup_steps"),
        "weight_decay": ("optimizer", "weight_decay"),
        "warmup_ratio": ("optimizer", "warmup_ratio"),
        "loss_type": ("loss", "primary"),
    }

    # Merge override options that were explicitly passed in the CLI
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            if key in cli_mapping:
                section, nested_key = cli_mapping[key]
                if section not in config:
                    config[section] = {}
                config[section][nested_key] = value
            else:
                config[key] = value

    # Ensure required configuration keys are present
    required_keys = [
        ("training", "seed"),
        ("data", "train_dir"),
        ("data", "val_dir"),
        ("training", "out_dir"),
        ("model", "n_states"),
        ("model", "quantizer"),
        ("training", "max_epochs"),
        ("training", "batch_size"),
        ("optimizer", "lr"),
    ]
    for section, key in required_keys:
        if section not in config or key not in config[section]:
            raise ValueError(f"Missing required configuration key: {section}.{key}")

    # Seed all sources of randomness
    L.seed_everything(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])

    # Load pair data
    train_data_raw = np.load(os.path.join(config["data"]["train_dir"], "data.npy"))
    x_train_raw = train_data_raw[:, :, 0]
    y_train_raw = train_data_raw[:, :, 1]

    val_data_raw = np.load(os.path.join(config["data"]["val_dir"], "data.npy"))
    x_val_raw = val_data_raw[:, :, 0]
    y_val_raw = val_data_raw[:, :, 1]

    # Create train dataset (fits scaler)
    train_dataset = PairDataset(
        x_train_raw,
        y_train_raw,
        descriptor_jitter_std=config["data"].get("descriptor_jitter_std", 0.0),
        seed=config["training"]["seed"],
    )
    mean = train_dataset.mean
    std = train_dataset.std

    # Create val dataset (uses train scaler, NO jitter)
    val_dataset = PairDataset(
        x_val_raw,
        y_val_raw,
        mean=mean,
        std=std,
        descriptor_jitter_std=0.0,
        seed=config["training"]["seed"],
    )

    # Alignment-aware batching when requested and metadata is available; else flat random.
    alignments_per_batch = config["data"].get("alignments_per_batch", 0)
    alignment_ids = _load_alignment_ids(config["data"]["train_dir"], len(x_train_raw))
    if alignments_per_batch > 0 and alignment_ids is not None:
        sampler = AlignmentBatchSampler(
            alignment_ids,
            batch_size=config["training"]["batch_size"],
            alignments_per_batch=alignments_per_batch,
            seed=config["training"]["seed"],
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    else:
        if alignments_per_batch > 0:
            print("alignment-aware sampling requested but no row-aligned metadata; using random.")
        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0
        )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0
    )

    # Initialize model
    input_dim = x_train_raw.shape[1]
    hidden_dim = config["model"].get("hidden_dim", 64)
    z_dim = config["model"].get("z_dim", 4)

    # Map the quantizer parameter to match model input 'quantizer_type'
    quantizer = config["model"]["quantizer"]
    quantizer_type = "vq" if quantizer in ("vq", "ema_vq") else quantizer

    # Build model using config parameters
    model = TdiV2Model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        n_states=config["model"]["n_states"],
        quantizer_type=quantizer_type,
        fsq_levels=config["model"].get("fsq_levels"),
        decay=config["model"].get("decay", 0.99),
        eps=config["model"].get("eps", 1e-5),
        commitment_cost=config.get("loss", {}).get("commitment_weight", 0.25),
        l2_normalize=config["model"].get("l2_normalize", True),
        min_count=config["model"].get("min_count", 1.0),
        replacement_warmup_steps=config["model"].get("replacement_warmup_steps", 500),
        lambda_usage=config.get("loss", {}).get("usage_weight", 0.0),
        lambda_contrast=config.get("loss", {}).get("contrastive_weight", 0.0),
        lambda_self=config.get("loss", {}).get("self_reconstruction_weight", 0.1),
        temperature=config.get("loss", {}).get("temperature", 0.1),
        lr=config["optimizer"]["lr"],
        weight_decay=config.get("optimizer", {}).get("weight_decay", 1e-4),
        warmup_ratio=config.get("optimizer", {}).get("warmup_ratio", 0.03),
        quantizer_warmup_epochs=config["training"].get("continuous_warmup_epochs", 0),
        aux_ramp_epochs=config["training"].get("aux_ramp_epochs", 0),
        loss_type=config.get("loss", {}).get("primary", "smooth_l1"),
        kmeans_init=config["model"].get("kmeans_init", False),
    )

    model_to_fit: L.LightningModule = model
    if config["training"].get("torch_compile", False):
        print("Compiling model with torch.compile...")
        compiled = torch.compile(model)
        if isinstance(compiled, L.LightningModule):
            model_to_fit = compiled

    # Callbacks for validation score selection
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_score:.2f}",
        dirpath=config["training"]["out_dir"],
    )
    early_stopping = EarlyStopping(monitor="val_score", mode="max", patience=10)

    # Lightning Trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision=config["training"].get("precision", "bf16-mixed"),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 4),
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model_to_fit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint
    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"Loading best checkpoint from {best_path}")
        best_model = TdiV2Model.load_from_checkpoint(best_path)
    else:
        print("No checkpoint saved, exporting current model.")
        best_model = model

    # Export best model
    best_model.export_model(config["training"]["out_dir"], mean=mean, std=std)
    print(f"Exported best model and scaler to {config['training']['out_dir']}")


if __name__ == "__main__":
    main()
