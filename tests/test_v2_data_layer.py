"""Unit tests for the tdi.data preprocessing pipeline and Phase 0 correctness fixes."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from tdi.data import build_features, validate_cigar, validate_dataset
from tdi.data.cigar import CigarValidationError
from tdi.data.config import load_config
from tdi.v2 import features
from tdi.v2.training_data import FEATURE_CACHE, extract_features

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


@pytest.mark.parametrize("cigar", ["10P_BAD_TRAILING_TEXT", "1P1X", "0P", "0M", ""])
def test_cigar_validator_rejects_malformed_or_zero_length(cigar: str) -> None:
    """Malformed, partially parseable, and zero-length CIGAR ops are rejected."""
    with pytest.raises(CigarValidationError):
        validate_cigar(cigar, n_ref=20, n_query=20)


def test_cigar_validator_rejects_cursor_walk_out_of_bounds() -> None:
    """Non-emitting operations must still respect total structure lengths."""
    with pytest.raises(CigarValidationError):
        validate_cigar("999M", n_ref=20, n_query=20)


def test_cigar_validator_accepts_no_p_cigar_empty_pairs() -> None:
    """Valid CIGARs without P pairs return an empty two-column pair array."""
    pairs = validate_cigar("10M5I3D", n_ref=13, n_query=15)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.int64


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


def test_config_rejects_unimplemented_sequence_delta_convention(tmp_path: Path) -> None:
    """Unsupported sequence-delta conventions fail instead of being mislabeled."""
    config_path = _make_dataset(tmp_path, "out_bad_delta")
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("features", {})["sequence_delta_convention"] = "i_minus_j"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="j_minus_i"):
        load_config(config_path)


def test_build_features_metadata_row_count_matches_arrays(tmp_path: Path) -> None:
    """Pair metadata row count equals the pairs array length, per split."""
    config_path = _make_dataset(tmp_path, "out_build")
    out_dir = build_features(config_path)

    train_pairs = np.load(out_dir / "train_x_raw.npy")
    import pandas as pd

    train_meta = pd.read_parquet(out_dir / "train_metadata.parquet")
    assert len(train_meta) == train_pairs.shape[0]
    assert train_pairs.shape[0] > 0
    # Lean-but-auditable columns exist (SCOP join present even when lookup is empty).
    for col in (
        "alignment_id",
        "split_group_source",
        "fold_source",
        "superfamily_source",
        "ca_dist_superposed",
    ):
        assert col in train_meta.columns
    # Redundant / dropped provenance columns are gone.
    for col in ("row_id", "source_pairfile_row", "source_is_forward", "family_source"):
        assert col not in train_meta.columns


def test_validate_dataset_summary_is_strict_json_serializable(tmp_path: Path) -> None:
    """Validation summaries use Python scalar counts, not NumPy scalar values."""
    config_path = _make_dataset(tmp_path, "out_validate")
    summary = validate_dataset(config_path)

    assert all(type(value) is int for value in summary.values())
    json.dumps(summary, indent=2, allow_nan=False)


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
        "train_x_raw.npy",
        "train_y_raw.npy",
        "val_x_raw.npy",
        "val_y_raw.npy",
        "scaler.npz",
        "train_metadata.parquet",
        "structures.parquet",
        "report.json",
        "train_skipped_alignments.tsv",
        "val_skipped_alignments.tsv",
        "report.md",
        "DATACARD.md",
    ):
        assert (out_dir / name).exists(), name
    # Per-split report duplicates are dropped in favor of the single joint report.json.
    assert not (out_dir / "train_report.json").exists()
    assert not (out_dir / "val_report.json").exists()


def test_full_report_flag_gates_histograms(tmp_path: Path) -> None:
    """Histograms are omitted by default and present (strict-JSON) with --full-report."""
    config_path = _make_dataset(tmp_path, "out_report_bins")

    # Default build: lean report, no histograms.
    out_default = build_features(config_path)
    default_report = json.loads((out_default / "report.json").read_text())
    assert "ca_distance_histogram" not in default_report["train"]
    assert "sequence_separation_histogram" not in default_report["train"]

    # --full-report build: histograms present with strict labeled bins.
    out_full = build_features(config_path, {"preprocessing.full_report": True}, force=True)
    full_report = json.loads((out_full / "report.json").read_text())
    json.dumps(full_report, indent=2, allow_nan=False)
    ca_hist = full_report["train"]["ca_distance_histogram"]
    assert set(ca_hist) == {"bins"}
    assert ca_hist["bins"][-1]["label"] == ">=5.0"
    assert all("count" in bin_record for bin_record in ca_hist["bins"])
    assert "sequence_separation_histogram" in full_report["train"]


def test_alignment_id_is_enriched_and_split_group_is_superfamily(tmp_path: Path) -> None:
    """alignment_id encodes pairfile stem + row + both sids; split_group == superfamily."""
    import pandas as pd

    lookup = tmp_path / "scop_lookup.tsv"
    lookup.write_text("d1aaaa_\ta.3.1.2\nd1bbbb_\tb.1.1.1\n")

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for sid_seed, sid in enumerate(("d1aaaa_", "d1bbbb_")):
        _write_pdb(pdb_dir / sid, seed=sid_seed + 1)
    pairs = tmp_path / "train_pairs.out"
    pairs.write_text("d1aaaa_ d1bbbb_ 24P\n")

    config = {
        "dataset": {
            "name": "synthetic",
            "pdb_dir": str(pdb_dir),
            "train_pairfile": str(pairs),
            "val_pairfile": str(pairs),
            "scop_lookup": str(lookup),
        },
        "features": {"virtual_center": [270.0, 0.0, 2.0], "max_ca_dist": 5.0},
        "sampling": {"max_pairs_per_alignment": None, "seed": 123},
        "outputs": {"out_dir": str(tmp_path / "out_align_id")},
    }
    config_path = tmp_path / "config_align_id.yaml"
    config_path.write_text(yaml.safe_dump(config))

    out_dir = build_features(config_path)
    meta = pd.read_parquet(out_dir / "train_metadata.parquet")
    assert (meta["alignment_id"] == "train_pairs:0:d1aaaa_:d1bbbb_").all()
    # split_group mirrors the superfamily-level grouping make_splits.py uses.
    assert (meta["split_group_source"] == meta["superfamily_source"]).all()
    forward = meta[meta["sid_source"] == "d1aaaa_"].iloc[0]
    assert forward["superfamily_source"] == "a.3.1"
    assert forward["fold_source"] == "a.3"


def test_too_few_pairs_build_drops_without_nan_metadata(tmp_path: Path) -> None:
    """Too-few-pair Kabsch inputs drop pairs instead of emitting NaN metadata rows."""
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for sid_seed, sid in enumerate(("d1aaaa_", "d1bbbb_")):
        _write_pdb(pdb_dir / sid, seed=sid_seed + 1)

    pairs = tmp_path / "pairs.out"
    pairs.write_text("d1aaaa_ d1bbbb_ 8M2P\n")

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
        "outputs": {"out_dir": str(tmp_path / "out_too_few")},
    }
    config_path = tmp_path / "config_too_few.yaml"
    config_path.write_text(yaml.safe_dump(config))

    out_dir = build_features(config_path)
    import pandas as pd

    train_pairs = np.load(out_dir / "train_x_raw.npy")
    train_meta = pd.read_parquet(out_dir / "train_metadata.parquet")
    report = json.loads((out_dir / "report.json").read_text())

    assert train_pairs.shape == (0, 10)
    assert list(train_meta.columns)
    assert "ca_dist_superposed" in train_meta.columns
    assert train_meta["ca_dist_superposed"].isna().sum() == 0
    assert report["train"]["stage_counts"]["n_alignments_dropped_degenerate_kabsch"] == 1
    assert report["train"]["stage_counts"]["n_pairs_dropped_degenerate_kabsch"] == 2


def test_build_features_skipped_alignments(tmp_path: Path) -> None:
    """Verify that a malformed/missing alignment is skipped, logged in the TSV,
    and does not crash the run.
    """
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    _write_pdb(pdb_dir / "d1aaaa_", seed=1)

    pairs = tmp_path / "pairs.out"
    pairs.write_text("d1aaaa_ d1bbbb_ 24P\n")

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
        "preprocessing": {
            "fail_on_skipped_alignments": False,
            "max_skipped_fraction": 1.0,
        },
        "outputs": {"out_dir": str(tmp_path / "out_skipped")},
    }
    config_path = tmp_path / "config_skipped.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    out_dir = build_features(config_path)
    tsv_path = out_dir / "train_skipped_alignments.tsv"
    assert tsv_path.exists()
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    assert len(df) == 1
    assert df.iloc[0]["sid1"] == "d1aaaa_"
    assert df.iloc[0]["sid2"] == "d1bbbb_"
    assert "FileNotFoundError" in str(df.iloc[0]["error_type"])


def test_build_features_fail_on_skipped_alignments(tmp_path: Path) -> None:
    """Verify that if fail_on_skipped_alignments is True, a skip raises RuntimeError."""
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    _write_pdb(pdb_dir / "d1aaaa_", seed=1)

    pairs = tmp_path / "pairs.out"
    pairs.write_text("d1aaaa_ d1bbbb_ 24P\n")

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
        "preprocessing": {
            "fail_on_skipped_alignments": True,
            "max_skipped_fraction": 0.01,
        },
        "outputs": {"out_dir": str(tmp_path / "out_failed")},
    }
    config_path = tmp_path / "config_failed.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    with pytest.raises(RuntimeError, match="fail_on_skipped_alignments is set"):
        build_features(config_path)


def test_parse_pairfile_line_resilience() -> None:
    """Verify that parse_pairfile_line correctly parses both 3-column and multi-column lines."""
    from tdi.v2.util import parse_pairfile_line

    # 3-column format
    res3 = parse_pairfile_line("d1aaaa_ d1bbbb_ 24P\n")
    assert res3 == ("d1aaaa_", "d1bbbb_", "24P")

    # Multi-column format (e.g. 10 columns)
    res10 = parse_pairfile_line("d12asa_ d1b8aa2 0.73 0.75 0.73 3.0 327 335 285 23I1M2P\n")
    assert res10 == ("d12asa_", "d1b8aa2", "23I1M2P")

    # Short line
    assert parse_pairfile_line("d1aaaa_ d1bbbb_") is None


def test_resolve_pdb_path_fallbacks(tmp_path: Path) -> None:
    """Verify resolve_pdb_path prioritizes no-extension then .pdb fallback."""
    from tdi.v2.util import resolve_pdb_path

    # Create dummy files
    dir_path = tmp_path / "structures"
    dir_path.mkdir()

    file_no_ext = dir_path / "d1aaaa_"
    file_no_ext.touch()

    file_with_ext = dir_path / "d1bbbb_.pdb"
    file_with_ext.touch()

    # Priority 1: no-extension exists
    assert resolve_pdb_path(dir_path, "d1aaaa_") == file_no_ext

    # Priority 2: no-extension missing, .pdb exists
    assert resolve_pdb_path(dir_path, "d1bbbb_") == file_with_ext

    # Missing case: returns path1 default (no extension)
    assert resolve_pdb_path(dir_path, "d1cccc_") == dir_path / "d1cccc_"
