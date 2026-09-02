# ruff: noqa: E402
"""Regression tests for archive extraction and normalized-structure refresh semantics."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.fetch_scop40_structures import safe_extract_tar, wrangle_structures


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_extraction_marker_tracks_archive_content(tmp_path: Path) -> None:
    """Replacing an archive refreshes extraction even without an explicit force flag."""
    archive = tmp_path / "structures.tar.gz"
    extracted = tmp_path / "extracted"
    _write_archive(archive, {"d1aaaa_": b"first"})

    assert safe_extract_tar(archive, extracted) is True
    assert safe_extract_tar(archive, extracted) is False
    assert (extracted / "d1aaaa_").read_bytes() == b"first"

    _write_archive(archive, {"d1bbbb_": b"second"})
    assert safe_extract_tar(archive, extracted) is True
    assert not (extracted / "d1aaaa_").exists()
    assert (extracted / "d1bbbb_").read_bytes() == b"second"


def test_wrangle_requires_force_before_overwriting_changed_structure(tmp_path: Path) -> None:
    """Without force, an existing normalized SID is reused only when content matches."""
    extracted = tmp_path / "extracted"
    normalized = tmp_path / "normalized"
    extracted.mkdir()
    source = extracted / "d1aaaa_"
    source.write_bytes(b"original")

    wrangle_structures(extracted, normalized)
    source.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="force-wrangle"):
        wrangle_structures(extracted, normalized)
    rows = wrangle_structures(extracted, normalized, force=True)
    assert len(rows) == 1
    assert (normalized / "d1aaaa_").read_bytes() == b"changed"
