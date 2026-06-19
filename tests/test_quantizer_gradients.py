import pytest
import torch

from tdi.v2.quantizer_gradients import (
    apply_quantizer_gradient,
    rotation_trick,
    straight_through_estimator,
)


def test_ste_forward_value_equals_quantized() -> None:
    """Verify that the straight-through estimator forward pass outputs exactly z_q."""
    z = torch.randn(8, 4, requires_grad=True)
    z_q = torch.randn(8, 4)
    out = straight_through_estimator(z, z_q)
    assert torch.allclose(out, z_q)


def test_rotation_trick_forward_value_equals_quantized() -> None:
    """Verify that the rotation trick forward pass outputs exactly z_q."""
    z = torch.randn(8, 4, requires_grad=True)
    z_q = torch.randn(8, 4)
    out = rotation_trick(z, z_q)
    assert torch.allclose(out, z_q)


def test_rotation_trick_backward_is_finite() -> None:
    """Verify that the rotation trick backpropagates finite gradients."""
    z = torch.randn(16, 4, requires_grad=True)
    z_q = torch.randn(16, 4)
    out = rotation_trick(z, z_q)
    loss = out.square().mean()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_rotation_trick_handles_zero_vectors() -> None:
    """Verify that the rotation trick handles zero inputs gracefully without NaN gradients."""
    z = torch.zeros(8, 4, requires_grad=True)
    z_q = torch.randn(8, 4)
    out = rotation_trick(z, z_q)
    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_apply_quantizer_gradient_modes() -> None:
    """Verify apply_quantizer_gradient returns correct forward values for all valid modes."""
    z = torch.randn(8, 4, requires_grad=True)
    z_q = torch.randn(8, 4)
    out_ste = apply_quantizer_gradient(z, z_q, mode="ste")
    out_rot = apply_quantizer_gradient(z, z_q, mode="rotation_trick")
    assert torch.allclose(out_ste, z_q)
    assert torch.allclose(out_rot, z_q)


def test_invalid_gradient_mode_raises() -> None:
    """Verify apply_quantizer_gradient raises ValueError for unsupported modes."""
    z = torch.randn(8, 4)
    z_q = torch.randn(8, 4)
    with pytest.raises(ValueError, match="Unknown quantizer gradient mode"):
        apply_quantizer_gradient(z, z_q, mode="bad_mode")
