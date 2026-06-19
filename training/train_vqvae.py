"""Training script for Vector Quantized Variational Autoencoder (VQ-VAE).

Learns 3Di alphabet states from aligned structural descriptors.
"""

import argparse
import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """Decoder module of VQ-VAE. Reconstructs features from quantized latents."""

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mu = nn.Linear(hidden_dim, input_dim)
        self.logvar = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)
        var = torch.exp(self.logvar(x))  # Predict variance directly
        return self.mu(x), var


class VectorQuantizer(nn.Module):
    """Vector Quantizer layer that maps continuous latent space to discrete embeddings."""

    def __init__(self, n_states: int, z_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_states, z_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_states, 1.0 / n_states)
        self.commitment_cost = 0.25
        self.n_states = n_states

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate distances between inputs and embeddings
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.weight.t())
        )

        # Retrieve nearest encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.n_states, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1.0)

        # Quantize latent space
        quantized = torch.matmul(encodings, self.embedding.weight)

        # Vector quantization losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator trick
        quantized = inputs + (quantized - inputs).detach()

        # Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return loss, quantized, perplexity, encodings


class VAE_VQ(nn.Module):
    """Full VQ-VAE model wrapping Encoder, Vector Quantizer, and Decoder."""

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: Decoder,
        z_dim: int,
        n_states: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = VectorQuantizer(n_states, z_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.vq(z)
        mu, var = self.decoder(quantized)
        return loss, (mu, var), perplexity, encodings


def batched(data: Tuple[np.ndarray, np.ndarray], batch_size: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition input tuple dataset into batches.

    Args:
        data: Tuple of (feat_x, feat_y) datasets.
        batch_size: If > 0, size of batch. Otherwise returns entire dataset as one batch.

    Returns:
        List of batched tuples.
    """
    if batch_size > 0:
        n_samples = len(data[0])
        return [
            (data[0][i * batch_size : (i + 1) * batch_size],
             data[1][i * batch_size : (i + 1) * batch_size])
            for i in range(n_samples // batch_size)
        ]
    return [data]


def create_vqvae(
    seed: int, input_dim: int, hidden_dim: int, z_dim: int, n_states: int
) -> VAE_VQ:
    """Instantiate and initialize VQ-VAE model with fixed random seed.

    Args:
        seed: Random seed.
        input_dim: Dimension of input features.
        hidden_dim: Dimension of internal hidden linear layers.
        z_dim: Embedding dimension.
        n_states: Number of discrete states in the alphabet.

    Returns:
        Configured VAE_VQ model.
    """
    torch.manual_seed(seed)
    encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, z_dim),
    )
    decoder = Decoder(input_dim, hidden_dim, z_dim)
    return VAE_VQ(encoder, decoder, z_dim, n_states)


def train_vqvae(
    model: VAE_VQ,
    training_data: Tuple[np.ndarray, np.ndarray],
    n_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> float:
    """Run backpropagation training on the VQ-VAE model.

    Args:
        model: VAE_VQ model.
        training_data: Tuple containing train features X and targets Y.
        n_epochs: Number of optimization epochs.
        lr: Optimizer learning rate.
        batch_size: Size of train batches.
        device: Target execution hardware device.

    Returns:
        Final training loss value.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.GaussianNLLLoss()
    model.train()

    last_loss = 0.0
    for epoch in range(n_epochs):
        batches = batched(training_data, batch_size)
        for feat_x, feat_y in batches:
            x_tensor = torch.tensor(feat_x, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(feat_y, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            vq_loss, (mu, var), _, _ = model(x_tensor)
            recon_loss = loss_fn(y_tensor, mu, var)

            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

    print(f"opt_loss= {last_loss:.3f}")
    return last_loss


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """Fuses parameters of a Linear layer and a BatchNorm1d layer for deployment.

    Args:
        linear: Source Linear layer.
        bn: Source BatchNorm1d layer.

    Returns:
        A new fused nn.Linear layer.
    """
    with torch.no_grad():
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias

        w = linear.weight
        b = linear.bias if linear.bias is not None else torch.zeros_like(mean)

        scale = gamma / torch.sqrt(var + eps)
        w_fused = w * scale.unsqueeze(1)
        b_fused = (b - mean) * scale + beta

        fused = nn.Linear(linear.in_features, linear.out_features)
        fused.weight.copy_(w_fused)
        fused.bias.copy_(b_fused)
        return fused


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
    parser.add_argument(
        "n_states", type=int, help="Size of discrete 3Di alphabet states."
    )
    args = parser.parse_args()

    # Load Data
    training_data_raw = np.load(args.data_path)
    training_data = (training_data_raw[:, :, 0], training_data_raw[:, :, 1])

    # Parameters
    input_dim = training_data[0].shape[1]
    hidden_dim = input_dim
    z_dim = 2
    batch_size = 512
    lr = 1e-3
    n_epochs = 4

    # Determine execution device
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

    # Move back to CPU for fusing and exporting parameters
    model.cpu()

    # Simplify encoder: fuse Linear and BatchNorm layers
    encoder_fused = nn.Sequential(
        fuse_linear_bn(model.encoder[0], model.encoder[1]),
        model.encoder[2],
        fuse_linear_bn(model.encoder[3], model.encoder[4]),
        model.encoder[5],
        model.encoder[6],
    )

    # Save exports
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(encoder_fused, os.path.join(args.out_dir, "encoder.pt"))
    torch.save(model.decoder, os.path.join(args.out_dir, "decoder.pt"))
    np.savetxt(
        os.path.join(args.out_dir, "states.txt"),
        model.vq.embedding.weight.detach().numpy(),
    )


if __name__ == "__main__":
    main()
