"""Unit tests for the tdi.data preprocessing pipeline and Phase 0 correctness fixes."""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from torch.utils.data import DataLoader

from tdi.data import build_features, validate_cigar
from tdi.data.cigar import CigarValidationError
from tdi.data.config import load_config
from tdi.v2 import features
from tdi.v2.training_data import FEATURE_CACHE, PairDataset, extract_features

_VIRT_A = (270.0, 0.0, 2.0)
_VIRT_B = (40.0, 0.0, 8.0)
_N_RES = 24


def _write_pdb(path: Path, n_res: int = _N_RES, seed: int = 1) -> None:
    """Write a synthetic single-chain ALA PDB (random-walk backbone) with full backbone + CB.

    Geometry is randomized (but seeded, so identical across runs) so that the virtual-CB
    move actually shifts nearest-residue partner choices.
    """
    rng = np.random.default_rng(seed)
    ca = np.cumsum(rng.standard_normal((n_res, 3)) * 3.0, axis=0)
    lines = []
    serial = 1
    for i in range(n_res):
        cx, cy, cz = ca[i]
        cb_dir = rng.standard_normal(3)
        cb_dir /= np.linalg.norm(cb_dir)
        atoms = {
            "N": (cx - 1.0, cy + 1.0, cz),
            "CA": (cx, cy, cz),
            "C": (cx + 1.0, cy + 1.0, cz),
            "O": (cx + 1.0, cy + 2.0, cz),
            "CB": tuple(np.array([cx, cy, cz]) + cb_dir * 1.5),
        }
        for name, (x, y, z) in atoms.items():
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
            )
            serial += 1
    lines.append("TER")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Phase 0 correctness fixes
# ---------------------------------------------------------------------------


def test_cache_key_disambiguates_virt_cb(tmp_path: Path) -> None:
    """Different virt_cb on the same PDB yields distinct features and cache entries."""
    pdb = tmp_path / "d_helix.pdb"
    _write_pdb(pdb)
    FEATURE_CACHE.clear()

    feat_a, _, _ = extract_features(str(pdb), _VIRT_A)
    feat_b, _, _ = extract_features(str(pdb), _VIRT_B)

    # Two distinct cache entries for the same path (old path-only key would collide).
    keys_for_path = [k for k in FEATURE_CACHE if k[0] == os.path.abspath(str(pdb))]
    assert len(keys_for_path) == 2

    # virt_cb A is still retrievable unchanged after the virt_cb B call (no stale overwrite).
    # Features carry NaN for boundary residues, so compare with NaNs zeroed.
    feat_a_again, _, _ = extract_features(str(pdb), _VIRT_A)
    assert np.array_equal(np.nan_to_num(feat_a), np.nan_to_num(feat_a_again))
    assert not np.array_equal(np.nan_to_num(feat_a), np.nan_to_num(feat_b))


def test_extract_features_caches_raw_coords(tmp_path: Path) -> None:
    """Cached coords are the raw parsed coords (CB not moved); input not mutated."""
    pdb = tmp_path / "d_helix.pdb"
    _write_pdb(pdb)
    FEATURE_CACHE.clear()

    parsed, _ = features.get_coords_from_pdb(str(pdb), full_backbone=True)
    _feat, _mask, raw = extract_features(str(pdb), _VIRT_A)

    # CA columns and CB columns equal the parser output (CB is NOT the move_CB result).
    assert np.allclose(raw[:, 0:3], parsed[:, 0:3], equal_nan=True)
    assert np.allclose(raw[:, 3:6], parsed[:, 3:6], equal_nan=True)


def test_move_cb_mutates_so_extract_must_copy(tmp_path: Path) -> None:
    """Document that move_CB mutates its input (justifying the .copy() in extract_features)."""
    pdb = tmp_path / "d_helix.pdb"
    _write_pdb(pdb)
    coords, _ = features.get_coords_from_pdb(str(pdb), full_backbone=True)
    before = coords.copy()
    features.move_CB(coords, virt_cb=_VIRT_A)
    assert not np.array_equal(coords[:, 3:6], before[:, 3:6])


def _concat_batches(loader: DataLoader) -> np.ndarray:
    xs = [x.numpy() for x, _y in loader]
    return np.concatenate(xs, axis=0)


def test_jitter_identical_across_num_workers() -> None:
    """Deterministic per-item jitter produces identical batches for num_workers in {0, 4}."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 10)).astype(np.float32)
    y = rng.standard_normal((40, 10)).astype(np.float32)

    def make_loader(num_workers: int) -> DataLoader:
        ds = PairDataset(x, y, descriptor_jitter_std=0.1, seed=7)
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=num_workers)

    out0 = _concat_batches(make_loader(0))
    out4 = _concat_batches(make_loader(4))
    assert np.array_equal(out0, out4)


# ---------------------------------------------------------------------------
# CIGAR validation (Task 1.6)
# ---------------------------------------------------------------------------


def test_cigar_validator_accepts_in_range() -> None:
    """A valid CIGAR within both structure ranges parses without error."""
    pairs = validate_cigar("5P", n_ref=5, n_query=5)
    assert pairs.shape == (5, 2)


def test_cigar_validator_rejects_out_of_range() -> None:
    """An aligned index beyond n_ref is rejected loudly."""
    with pytest.raises(CigarValidationError):
        validate_cigar("5P", n_ref=3, n_query=5)


# ---------------------------------------------------------------------------
# Config + full build (Tasks 1.1, 1.3, 1.7)
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path: Path, out_name: str) -> Path:
    """Write synthetic PDBs + pairfiles + YAML config; return the config path."""
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for sid_seed, sid in enumerate(("d1aaaa_", "d1bbbb_")):
        _write_pdb(pdb_dir / sid, seed=sid_seed + 1)

    pairs = tmp_path / "pairs.out"
    # Self-alignments of 24 residues guarantee surviving pairs after filters.
    pairs.write_text("d1aaaa_ d1bbbb_ 24P\nd1bbbb_ d1aaaa_ 24P\n")

    config = {
        "dataset": {
            "name": "synthetic",
            "pdb_dir": str(pdb_dir),
            "train_pairfile": str(pairs),
            "val_pairfile": str(pairs),
            "scop_lookup": None,
        },
        "features": {"virtual_center": [270.0, 0.0, 2.0], "max_ca_dist": 5.0},
        "sampling": {"max_pairs_per_alignment": None, "seed": 123},
        "outputs": {"out_dir": str(tmp_path / out_name)},
    }
    config_path = tmp_path / f"config_{out_name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_config_roundtrip_and_hash(tmp_path: Path) -> None:
    """Config loads, and the content hash is stable + override-sensitive."""
    config_path = _make_dataset(tmp_path, "out_cfg")
    cfg = load_config(config_path)
    assert cfg.dataset.name == "synthetic"
    assert cfg.features.virtual_center == (270.0, 0.0, 2.0)
    h1 = cfg.config_hash()
    cfg2 = load_config(config_path, {"sampling.seed": 999})
    assert cfg2.sampling.seed == 999
    assert cfg2.config_hash() != h1


def test_build_features_metadata_row_count_matches_arrays(tmp_path: Path) -> None:
    """Pair metadata row count equals the pairs array length, per split."""
    config_path = _make_dataset(tmp_path, "out_build")
    out_dir = build_features(config_path)

    train_pairs = np.load(out_dir / "train_pairs.npy")
    import pandas as pd

    train_meta = pd.read_parquet(out_dir / "train_metadata.parquet")
    assert len(train_meta) == train_pairs.shape[0]
    assert train_pairs.shape[0] > 0
    # Expected metadata columns exist (SCOP join present even when lookup is empty).
    for col in ("row_id", "alignment_id", "fold_source", "ca_dist_superposed"):
        assert col in train_meta.columns


def test_build_refuses_overwrite(tmp_path: Path) -> None:
    """A populated out_dir is immutable unless force=True."""
    config_path = _make_dataset(tmp_path, "out_immut")
    build_features(config_path)
    with pytest.raises(FileExistsError):
        build_features(config_path)


def test_build_is_deterministic(tmp_path: Path) -> None:
    """Two runs on identical inputs produce identical output array hashes."""
    import json

    cfg1 = _make_dataset(tmp_path, "out_run1")
    cfg2 = _make_dataset(tmp_path, "out_run2")
    out1 = build_features(cfg1)
    out2 = build_features(cfg2)

    man1 = json.loads((out1 / "manifest.json").read_text())
    man2 = json.loads((out2 / "manifest.json").read_text())
    hashes1 = {k: v["sha256"] for k, v in man1["outputs"].items()}
    hashes2 = {k: v["sha256"] for k, v in man2["outputs"].items()}
    assert hashes1 == hashes2


def test_build_writes_expected_artifacts(tmp_path: Path) -> None:
    """All immutable-layout artifacts are produced."""
    config_path = _make_dataset(tmp_path, "out_artifacts")
    out_dir = build_features(config_path)
    for name in (
        "manifest.json",
        "train_pairs.npy",
        "val_pairs.npy",
        "scaler.npz",
        "train_metadata.parquet",
        "structures.parquet",
        "report.json",
        "report.md",
        "DATACARD.md",
    ):
        assert (out_dir / name).exists(), name
