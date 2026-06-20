"""Surrogate-gradient estimators for the vector-quantization bottleneck.

Provides the standard straight-through estimator and the rotation trick
(Fifty et al., 2024), selected via :func:`apply_quantizer_gradient`.
"""

import torch
import torch.nn.functional as F


def straight_through_estimator(z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """Standard VQ straight-through estimator.

    Forward:
        Returns the quantized representation z_q.
    Backward:
        Gradients dL/dz are copied directly from dL/dz_q,
        bypassing/ignoring the non-differentiable quantization operation.

    Args:
        z: Continuous latent tensor from encoder.
        z_q: Quantized latent tensor from codebook.

    Returns:
        Tensor with the forward value of z_q and the gradient path of z.
    """
    # Detach (z_q - z) so that backpropagation flows through z only.
    return z + (z_q - z).detach()


def rotation_trick(
    z: torch.Tensor,
    z_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rotation-trick surrogate gradient for vector quantization.

    Forward:
        Exactly returns the quantized representation z_q.
    Backward:
        Gradients to z are rotated/scaled to align with the chosen
        codebook vector z_q, preserving the angular relationship.

    We construct the minimal Householder reflection that maps z's unit
    direction to z_q's unit direction, then apply it to transform
    backpropagated gradients. Following the published method, the rotation
    basis and the magnitude rescale are treated as constants (stop-gradient),
    so ``z_rot`` is a constant linear map ``R @ z`` and the encoder gradient is
    ``R^T g``. Only detaching the basis (not ``z`` itself) keeps the map linear.

    Args:
        z: Continuous latent tensor from encoder of shape (..., dim).
        z_q: Quantized latent tensor from codebook of shape (..., dim).
        eps: Epsilon value to prevent division by zero in normalize/norm functions.

    Returns:
        Tensor with forward value of z_q and custom backward flow.
    """
    if z.shape != z_q.shape:
        raise ValueError(f"z and z_q must have same shape, got {z.shape} and {z_q.shape}")

    original_dtype = z.dtype
    # Work in float32 for numerical stability across different precisions/autocast
    z_f = z.float()
    z_q_f = z_q.float()

    # Rotation basis is a constant of the transform: detach so gradients do not leak
    # through the normalize/bisector construction (R must be treated as constant).
    z_norm = F.normalize(z_f, dim=-1, eps=eps).detach()
    zq_norm = F.normalize(z_q_f, dim=-1, eps=eps).detach()

    # Householder vector for rotating z_norm toward zq_norm.
    # v = z_norm + zq_norm defines the bisector reflection plane (detached basis).
    v = F.normalize(z_norm + zq_norm, dim=-1, eps=eps)

    # Householder reflection: R(z) = 2 v (v^T z) - z, with v constant. This is linear in
    # z, so autograd yields R^T on the backward pass (the rotation-trick gradient).
    dot = torch.sum(z_f * v, dim=-1, keepdim=True)
    z_rot = 2.0 * v * dot - z_f

    # Match the magnitude of z_q; the rescale factor is a constant (detached).
    z_q_norm_mag = torch.linalg.vector_norm(z_q_f, dim=-1, keepdim=True).clamp_min(eps)
    z_rot_norm_mag = torch.linalg.vector_norm(z_rot, dim=-1, keepdim=True).clamp_min(eps)
    z_rot = z_rot * (z_q_norm_mag / z_rot_norm_mag).detach()

    # Cast back to original input dtype
    z_rot = z_rot.to(original_dtype)

    # Forward: return z_q. Backward: flow through z_rot.
    return z_rot + (z_q - z_rot).detach()


def apply_quantizer_gradient(
    z: torch.Tensor,
    z_q: torch.Tensor,
    mode: str = "rotation_trick",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply the selected quantizer surrogate-gradient path.

    Args:
        z: Continuous latent tensor from encoder.
        z_q: Quantized latent tensor.
        mode: Gradient estimator mode ("ste" or "rotation_trick").
        eps: Epsilon scalar.

    Returns:
        Tensor with quantized values routed through chosen gradient estimator path.
    """
    if mode == "ste":
        return straight_through_estimator(z, z_q)
    if mode == "rotation_trick":
        return rotation_trick(z, z_q, eps=eps)
    raise ValueError(f"Unknown quantizer gradient mode: {mode!r}")
