# ruff: noqa: E402
"""Smoke/property tests for the self-contained quarantined experiments.

These keep the reintroduced snapshots runnable (not dead code). They import from the top-level
``experiments`` package and exercise the contract of each removed-but-self-contained feature.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import torch

from experiments.alignment_batch_sampler import AlignmentBatchSampler
from experiments.augmentation import jitter_coords
from experiments.rotation_trick import apply_quantizer_gradient, rotation_trick


def test_rotation_trick_forward_is_zq_backward_flows_to_z() -> None:
    """Forward returns z_q exactly; gradient reaches z (custom backward)."""
    z = torch.randn(8, 4, requires_grad=True)
    z_q = torch.randn(8, 4)
    out = rotation_trick(z, z_q)
    assert torch.allclose(out, z_q, atol=1e-5)
    out.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_apply_quantizer_gradient_modes() -> None:
    """Both modes return z_q in the forward and reject unknown modes."""
    z = torch.randn(4, 3, requires_grad=True)
    z_q = torch.randn(4, 3)
    for mode in ("ste", "rotation_trick"):
        assert torch.allclose(apply_quantizer_gradient(z, z_q, mode=mode), z_q, atol=1e-5)
    with pytest.raises(ValueError, match="Unknown quantizer gradient mode"):
        apply_quantizer_gradient(z, z_q, mode="nope")


def test_jitter_coords_masks_and_is_reproducible() -> None:
    """Jitter touches only valid rows and is reproducible under a fixed seed."""
    coords = np.zeros((6, 3), dtype=np.float32)
    mask = np.array([True, True, False, True, False, True])
    out1 = jitter_coords(coords, mask, std=0.5, rng=np.random.default_rng(0))
    out2 = jitter_coords(coords, mask, std=0.5, rng=np.random.default_rng(0))
    assert np.array_equal(out1, out2)
    assert np.all(out1[~mask] == 0.0)
    assert not np.all(out1[mask] == 0.0)


def test_alignment_batch_sampler_spans_many_alignments() -> None:
    """Each batch spans >= alignments_per_batch distinct alignments, reproducibly."""
    alignment_ids = np.repeat(np.arange(20), 50)
    sampler = AlignmentBatchSampler(alignment_ids, batch_size=64, alignments_per_batch=8, seed=0)
    batches = list(sampler)
    assert len(batches) == len(alignment_ids) // 64
    for batch in batches:
        assert len({int(alignment_ids[i]) for i in batch}) >= 8
    sampler2 = AlignmentBatchSampler(alignment_ids, batch_size=64, alignments_per_batch=8, seed=0)
    assert list(sampler2) == batches
