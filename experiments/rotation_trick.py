"""Quarantined experiment: surrogate-gradient estimators for the VQ bottleneck.

Removed from the core (which now uses the straight-through estimator inline), this keeps the
rotation trick (Fifty et al., 2024) as a self-contained, runnable snapshot. It is pure-torch
and depends on nothing in ``tdi.v2``; drop ``apply_quantizer_gradient(z, z_q, mode=...)`` into a
quantizer's forward to reproduce the pre-refactor gradient routing.
"""

import torch
import torch.nn.functional as F


def straight_through_estimator(z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """Standard VQ straight-through estimator.

    Forward returns ``z_q``; backward copies ``dL/dz_q`` straight to ``z`` (bypassing the
    non-differentiable quantization).
    """
    return z + (z_q - z).detach()


def rotation_trick(z: torch.Tensor, z_q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation-trick surrogate gradient for vector quantization.

    Forward returns ``z_q`` exactly; backward rotates/scales gradients to ``z`` to align with
    the chosen codebook vector, preserving the angular relationship. The rotation basis and the
    magnitude rescale are treated as constants (stop-gradient), so ``z_rot`` is a constant linear
    map ``R @ z`` and the encoder gradient is ``R^T g``.

    Args:
        z: Continuous latent tensor from encoder of shape (..., dim).
        z_q: Quantized latent tensor from codebook of shape (..., dim).
        eps: Epsilon to prevent division by zero in normalize/norm.

    Returns:
        Tensor with forward value of z_q and custom backward flow.
    """
    if z.shape != z_q.shape:
        raise ValueError(f"z and z_q must have same shape, got {z.shape} and {z_q.shape}")

    original_dtype = z.dtype
    z_f = z.float()
    z_q_f = z_q.float()

    # Rotation basis is a constant of the transform: detach so gradients do not leak through
    # the normalize/bisector construction (R must be treated as constant).
    z_norm = F.normalize(z_f, dim=-1, eps=eps).detach()
    zq_norm = F.normalize(z_q_f, dim=-1, eps=eps).detach()

    # Householder vector for rotating z_norm toward zq_norm. Near-antipodal case (z_q ~= -z):
    # z_norm + zq_norm -> 0, so v is ill-conditioned; eps keeps normalize from NaN but the
    # reflection direction is then arbitrary (rare; left as-is in this snapshot).
    v = F.normalize(z_norm + zq_norm, dim=-1, eps=eps)

    # Householder reflection R(z) = 2 v (v^T z) - z, with v constant (linear in z; autograd
    # yields R^T on the backward pass).
    dot = torch.sum(z_f * v, dim=-1, keepdim=True)
    z_rot = 2.0 * v * dot - z_f

    # Match the magnitude of z_q; the rescale factor is a constant (detached).
    z_q_norm_mag = torch.linalg.vector_norm(z_q_f, dim=-1, keepdim=True).clamp_min(eps)
    z_rot_norm_mag = torch.linalg.vector_norm(z_rot, dim=-1, keepdim=True).clamp_min(eps)
    z_rot = z_rot * (z_q_norm_mag / z_rot_norm_mag).detach()

    z_rot = z_rot.to(original_dtype)
    # Forward: z_q. Backward: flow through z_rot.
    return z_rot + (z_q - z_rot).detach()


def apply_quantizer_gradient(
    z: torch.Tensor, z_q: torch.Tensor, mode: str = "rotation_trick", eps: float = 1e-8
) -> torch.Tensor:
    """Apply the selected quantizer surrogate-gradient path (``"ste"`` or ``"rotation_trick"``)."""
    if mode == "ste":
        return straight_through_estimator(z, z_q)
    if mode == "rotation_trick":
        return rotation_trick(z, z_q, eps=eps)
    raise ValueError(f"Unknown quantizer gradient mode: {mode!r}")
