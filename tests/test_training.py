"""Unit tests for the modernized 3Di VAE training pipeline components."""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure the training directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../training")))

import util
import extract_pdb_features
import train_vqvae


def test_parse_cigar() -> None:
    """Test parsing CIGAR strings to align residues."""
    # Action 'P' is a perfect match and indexes should align
    # M, D, and I are not returned as index mappings for substitution matrices
    cigar1 = "2P"
    res1 = util.parse_cigar(cigar1)
    assert np.array_equal(res1, np.array([[0, 0], [1, 1]]))

    cigar2 = "1M1D1I2P"
    # 1M: ref=1, query=1
    # 1D: ref=2, query=1
    # 1I: ref=2, query=2
    # 2P: ref advances 2, query advances 2
    # matches: (2, 2), (3, 3)
    res2 = util.parse_cigar(cigar2)
    assert np.array_equal(res2, np.array([[2, 2], [3, 3]]))


def test_mutual_information() -> None:
    """Test mutual information calculation."""
    # Joint distribution matrix
    p_ab = np.array([[0.25, 0.0], [0.0, 0.75]])
    mi = util.mutual_information(p_ab)
    # MI = sum( p(x,y) * log2( p(x,y) / (p(x)p(y)) ) )
    # p(0) = 0.25, p(1) = 0.75
    # q(0) = 0.25, q(1) = 0.75
    # MI = 0.25 * log2( 0.25 / (0.25 * 0.25) ) + 0.75 * log2( 0.75 / (0.75 * 0.75) )
    #    = 0.25 * log2(4) + 0.75 * log2(1.333...)
    #    = 0.25 * 2 + 0.75 * (log2(4) - log2(3)) = 0.5 + 1.5 - 0.75*log2(3) = 2 - 0.75 * 1.58496 = 2 - 1.1887 = 0.811
    assert np.isclose(mi, 0.811278, atol=1e-4)


def test_approx_c_beta_position() -> None:
    """Test approximation of C_beta coordinates."""
    c_alpha = np.array([0.0, 0.0, 0.0])
    n = np.array([1.0, 0.0, 0.0])
    c_carboxyl = np.array([0.0, 1.0, 0.0])
    c_beta = extract_pdb_features.approx_c_beta_position(c_alpha, n, c_carboxyl)
    assert c_beta.shape == (3,)
    assert not np.isnan(c_beta).any()


def test_distance_matrix() -> None:
    """Test coordinate distance matrix generation."""
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    dist = extract_pdb_features.distance_matrix(a, b)
    expected = np.array([[0.0, 2.0], [1.0, np.sqrt(5.0)]])
    assert np.allclose(dist, expected)


def test_create_vqvae() -> None:
    """Test VQ-VAE construction and parameters."""
    model = train_vqvae.create_vqvae(
        seed=42, input_dim=10, hidden_dim=10, z_dim=2, n_states=20
    )
    assert isinstance(model, train_vqvae.VAE_VQ)
    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, train_vqvae.Decoder)
    assert isinstance(model.vq, train_vqvae.VectorQuantizer)
    assert model.vq.n_states == 20


def test_fuse_linear_bn() -> None:
    """Test conversion and fusing of Linear and BatchNorm layers."""
    linear = nn.Linear(5, 10)
    bn = nn.BatchNorm1d(10)
    # Initialize mock batchnorm parameters
    bn.running_mean.fill_(0.5)
    bn.running_var.fill_(1.2)
    bn.weight.data.fill_(0.8)
    bn.bias.data.fill_(-0.2)

    # Fuse layer parameters
    fused = train_vqvae.fuse_linear_bn(linear, bn)
    assert isinstance(fused, nn.Linear)
    assert fused.in_features == 5
    assert fused.out_features == 10

    # Ensure fused outputs are equivalent on a sample input
    x = torch.randn(3, 5)
    with torch.no_grad():
        linear.eval()
        bn.eval()
        expected = bn(linear(x))
        fused.eval()
        output = fused(x)
        assert torch.allclose(output, expected, atol=1e-5)
