"""Vector Quantized Variational Autoencoder (VQ-VAE) model for learning 3Di alphabet states."""

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)
        var = torch.exp(self.logvar(x))
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_states, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1.0)

        quantized = torch.matmul(encodings, self.embedding.weight)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.vq(z)
        mu, var = self.decoder(quantized)
        return loss, (mu, var), perplexity, encodings


def batched(
    data: tuple[np.ndarray, np.ndarray], batch_size: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
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
            (
                data[0][i * batch_size : (i + 1) * batch_size],
                data[1][i * batch_size : (i + 1) * batch_size],
            )
            for i in range(n_samples // batch_size)
        ]
    return [data]


def create_vqvae(seed: int, input_dim: int, hidden_dim: int, z_dim: int, n_states: int) -> VAE_VQ:
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
    training_data: tuple[np.ndarray, np.ndarray],
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
    """Fuse parameters of a Linear layer and a BatchNorm1d layer for deployment.

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
        assert mean is not None
        assert var is not None
        assert gamma is not None
        assert beta is not None

        w = linear.weight
        b = linear.bias if linear.bias is not None else torch.zeros_like(mean)

        scale = gamma / torch.sqrt(var + eps)
        w_fused = w * scale.unsqueeze(1)
        b_fused = (b - mean) * scale + beta

        fused = nn.Linear(linear.in_features, linear.out_features)
        fused.weight.copy_(w_fused)
        fused.bias.copy_(b_fused)
        return fused
