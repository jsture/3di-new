#!/usr/bin/env python3
"""CLI: train VQ-VAE model to learn discrete 3Di state representations."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from tdi.model import create_vqvae, fuse_linear_bn, train_vqvae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train VQ-VAE model to learn discrete 3Di state representations."
    )
    parser.add_argument(
        "seed", type=int, help="Seed value for weight initialization reproducibility."
    )
    parser.add_argument("data_path", type=str, help="Path to training data .npy file.")
    parser.add_argument(
        "out_dir", type=str, help="Output directory to save trained model parameters."
    )
    parser.add_argument("n_states", type=int, help="Size of discrete 3Di alphabet states.")
    args = parser.parse_args()

    training_data_raw = np.load(args.data_path)
    training_data = (training_data_raw[:, :, 0], training_data_raw[:, :, 1])

    input_dim = training_data[0].shape[1]
    hidden_dim = input_dim
    z_dim = 2
    batch_size = 512
    lr = 1e-3
    n_epochs = 4

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Training on device: {device}")

    model = create_vqvae(args.seed, input_dim, hidden_dim, z_dim, args.n_states)
    model.to(device)

    train_vqvae(model, training_data, n_epochs, lr, batch_size, device)
    model.eval()
    model.cpu()

    encoder_fused = nn.Sequential(
        fuse_linear_bn(model.encoder[0], model.encoder[1]),
        model.encoder[2],
        fuse_linear_bn(model.encoder[3], model.encoder[4]),
        model.encoder[5],
        model.encoder[6],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(encoder_fused, os.path.join(args.out_dir, "encoder.pt"))
    torch.save(model.decoder, os.path.join(args.out_dir, "decoder.pt"))
    np.savetxt(
        os.path.join(args.out_dir, "states.txt"),
        model.vq.embedding.weight.detach().numpy(),
    )


if __name__ == "__main__":
    main()
