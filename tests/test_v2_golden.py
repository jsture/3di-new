"""Behavior-lock golden tests for the v2 data path (Phase 0 of the simplification plan).

These pin the invariants of the data pipeline (CIGAR parsing, feature extraction, Ca
superposition filtering, bidirectional alignment, the train-only scaler) plus the two
end-to-end contracts that survive the refactor: an exported model encodes states strictly
inside ``[0, n_states)``, and ``evaluate`` writes ``sequences.txt`` / ``submat.txt`` /
``evaluation_report.json``. They must stay green through every later phase, so intentional
simplification stays distinguishable from silent drift.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from tdi.v2.cli import run_evaluate
from tdi.v2.model import AlphabetModel
from tdi.v2.training_data import (
    align_features,
    extract_features,
    filter_ca_distance,
    fit_standardizer,
    transform,
)
from tdi.v2.util import parse_cigar

VIRT: tuple[float, float, float] = (0.0, 0.0, 1.0)


def _residue_atoms(i: int) -> dict[str, np.ndarray]:
    """Backbone + CB coordinates for residue ``i`` along a gentle 3D curve.

    The curve keeps consecutive CAs distinct and non-collinear so neighbor search and the
    Kabsch superposition (rank >= 2) are both well-defined.
    """
    ca = np.array([1.5 * i, 2.0 * np.sin(0.5 * i), 2.0 * np.cos(0.5 * i)], dtype=np.float32)
    n = ca + np.array([0.6, 0.5, 0.0], dtype=np.float32)
    c = ca + np.array([0.6, -0.5, 0.2], dtype=np.float32)
    cb = ca + np.array([0.0, 0.3, 1.2], dtype=np.float32)
    return {"N": n, "CA": ca, "C": c, "CB": cb}


def _write_pdb(path: Path, n_res: int = 10) -> None:
    """Write a valid multi-residue single-chain PDB via Biopython's writer."""
    structure = Structure("s")
    model = Model(0)
    chain = Chain("A")
    serial = 1
    for i in range(n_res):
        res = Residue((" ", i + 1, " "), "ALA", " ")
        for name, coord in _residue_atoms(i).items():
            atom = Atom(name, coord, 0.0, 1.0, " ", f" {name} ", serial, element=name[0])
            res.add(atom)
            serial += 1
        chain.add(res)
    model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


# ---------------------------------------------------------------------------
# 1. CIGAR parsing
# ---------------------------------------------------------------------------


def test_parse_cigar_rejects_unsupported_ops() -> None:
    """Unsupported ops raise rather than silently desynchronizing ref/query indices."""
    with pytest.raises(ValueError, match="not supported"):
        parse_cigar("3P2X")


def test_parse_cigar_parses_perfect_match_pairs() -> None:
    """``P`` records aligned pairs as an (N, 2) ref/query index map; M/D/I only advance."""
    pairs = parse_cigar("2M3P")
    assert pairs.shape == (3, 2)
    # 2M advances both ref and query to 2 before the 3P pairs begin.
    assert np.array_equal(pairs, np.array([[2, 2], [3, 3], [4, 4]]))


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------


def test_extract_features_shape_and_finite_valid_mask(tmp_path: Path) -> None:
    """``extract_features`` returns (N, 10) features and a finite-only valid mask."""
    pdb = tmp_path / "s.pdb"
    _write_pdb(pdb, n_res=10)

    feat, valid_mask, coords = extract_features(str(pdb), VIRT)

    assert feat.shape == (10, 10)
    assert valid_mask.shape == (10,)
    assert valid_mask.any()
    # The valid mask marks exactly the finite-feature rows.
    assert np.array_equal(valid_mask, np.isfinite(feat).all(axis=1))
    # Boundary residues never have both sequence neighbors, so they are invalid.
    assert not valid_mask[0]
    assert not valid_mask[-1]
    assert coords.shape == (10, 12)


# ---------------------------------------------------------------------------
# 3. Ca-distance superposition filter (CA columns 0:3 only)
# ---------------------------------------------------------------------------


def test_filter_ca_distance_superposes_and_uses_ca_columns_only() -> None:
    """A rigid-body-moved copy superposes to ~0 distance; only CA columns 0:3 matter."""
    ca = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rot_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    coords1 = np.zeros((4, 12))
    coords2 = np.zeros((4, 12))
    coords1[:, 0:3] = ca
    coords2[:, 0:3] = ca @ rot_z.T + np.array([10.0, -5.0, 2.0])
    # Garbage in the non-CA columns must not affect the result (CA-only contract).
    coords1[:, 3:12] = 99.0
    coords2[:, 3:12] = -42.0

    idx = np.array([0, 1, 2, 3])
    v1, v2, dists, error = filter_ca_distance(idx, idx, coords1, coords2, max_ca_dist=1e-3)

    assert error is None
    assert np.array_equal(v1, idx)
    assert np.array_equal(v2, idx)
    assert dists is not None
    assert np.all(dists < 1e-4)


def test_filter_ca_distance_removes_bad_pair_after_superposition() -> None:
    """A single displaced residue is dropped by the max_ca_dist filter post-superposition."""
    ca = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    moved = ca.copy()
    moved[2] += np.array([5.0, 0.0, 0.0])  # break residue 2 only

    coords1 = np.zeros((4, 12))
    coords2 = np.zeros((4, 12))
    coords1[:, 0:3] = ca
    coords2[:, 0:3] = moved

    idx = np.array([0, 1, 2, 3])
    v1, _v2, dists, error = filter_ca_distance(idx, idx, coords1, coords2, max_ca_dist=1.0)

    assert error is None
    assert 2 not in v1.tolist()
    assert dists is not None
    assert np.all(dists <= 1.0)


# ---------------------------------------------------------------------------
# 4. Bidirectional alignment ordering
# ---------------------------------------------------------------------------


def test_align_features_bidirectional_forward_then_reverse(tmp_path: Path) -> None:
    """``align_features`` emits forward (sid1->sid2) then reverse rows; meta length matches."""
    _write_pdb(tmp_path / "alpha.pdb", n_res=10)
    _write_pdb(tmp_path / "beta.pdb", n_res=10)

    x, y, meta = align_features(str(tmp_path), VIRT, "alpha", "beta", "10P")

    assert x.shape[1] == 10
    assert y.shape[1] == 10
    n = len(x)
    assert n > 0
    assert len(y) == n
    assert len(meta["sid_source"]) == n
    assert len(meta["sid_target"]) == n
    assert len(meta["idx_source"]) == n

    half = n // 2
    # Forward half then reverse half.
    assert all(s == "alpha" for s in meta["sid_source"][:half])
    assert all(s == "beta" for s in meta["sid_source"][half:])
    assert all(s == "beta" for s in meta["sid_target"][:half])
    assert all(s == "alpha" for s in meta["sid_target"][half:])


# ---------------------------------------------------------------------------
# 5. Train-only scaler (no val leakage)
# ---------------------------------------------------------------------------


def test_scaler_is_fit_on_train_and_reused_for_val() -> None:
    """``fit_standardizer`` reads only the array it is given; val is transformed with it."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(loc=3.0, scale=2.0, size=(200, 10)).astype(np.float32)
    x_val = rng.normal(loc=-5.0, scale=7.0, size=(50, 10)).astype(np.float32)

    mean, std = fit_standardizer(x_train)
    # Statistics derive from train alone (val's very different stats must not leak in).
    assert np.allclose(mean, x_train.mean(axis=0), atol=1e-4)
    assert np.allclose(std, np.maximum(x_train.std(axis=0), 1e-6), atol=1e-4)

    # Val standardized with the train scaler keeps val's own (non-zero-mean) distribution.
    val_scaled = transform(x_val, mean, std)
    assert not np.allclose(val_scaled.mean(axis=0), 0.0, atol=0.5)


# ---------------------------------------------------------------------------
# 6. Exported model encodes states inside [0, n_states)
# ---------------------------------------------------------------------------


def test_exported_model_encodes_states_in_range(tmp_path: Path) -> None:
    """A round-tripped export assigns every residue a state strictly in ``[0, n_states)``."""
    model = AlphabetModel(quantizer="vq", n_states=20, z_dim=4)
    model.save(tmp_path, mean=np.zeros(10), std=np.ones(10))
    loaded, _, _ = AlphabetModel.load(tmp_path)
    loaded.eval()

    x = torch.randn(64, 10)
    states = loaded.encode_states(x)
    assert states.shape == (64,)
    assert int(states.min()) >= 0
    assert int(states.max()) < loaded.n_states


# ---------------------------------------------------------------------------
# 7. evaluate writes the three artifacts
# ---------------------------------------------------------------------------


def test_evaluate_writes_three_artifacts(tmp_path: Path) -> None:
    """``evaluate`` produces sequences.txt, submat.txt, and evaluation_report.json."""
    model = AlphabetModel(quantizer="fsq", levels=[5, 4])
    model_dir = tmp_path / "model"
    model.save(model_dir, mean=np.zeros(10), std=np.ones(10))

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_pdb(pdb_dir / "alpha.pdb", n_res=10)
    _write_pdb(pdb_dir / "beta.pdb", n_res=10)

    pairfile = tmp_path / "pairs.txt"
    pairfile.write_text("alpha beta 10P\n")

    out_dir = tmp_path / "eval"
    args = SimpleNamespace(
        model_dir=str(model_dir),
        pdb_dir=str(pdb_dir),
        pairfile=str(pairfile),
        out_dir=str(out_dir),
        virt=[0.0, 0.0, 1.0],
        invalid_state="X",
    )
    run_evaluate(args)

    assert (out_dir / "sequences.txt").exists()
    assert (out_dir / "submat.txt").exists()
    assert (out_dir / "evaluation_report.json").exists()
