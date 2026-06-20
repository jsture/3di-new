"""Structure-level QC: one row per domain describing parse outcome and completeness.

Makes chain selection explicit (records that the first chain was used and how many
existed) so domains that yield no features are explainable from the table.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser

from tdi.data.hashing import sha256_file
from tdi.v2.features import get_atom_coordinates
from tdi.v2.util import resolve_pdb_path

# Module-level cache dict (keyed on path, mtime, size) to avoid redundant PDB parses
_STRUCTURE_QC_CACHE: dict[tuple[str, float, int], dict[str, object]] = {}


def structure_qc(sid: str, pdb_path: str | Path) -> dict[str, object]:
    """Compute a QC record for one structure.

    Args:
        sid: Structure id.
        pdb_path: Path to the PDB file.

    Returns:
        Dict with parse status and completeness fields. Never raises: parse failures
        are recorded as ``parse_status="parse_error"``.
    """
    path = str(pdb_path)
    record: dict[str, object] = {
        "sid": sid,
        "path": path,
        "n_residues": 0,
        "n_valid_residues": 0,
        "valid_fraction": 0.0,
        "n_chains": 0,
        "selected_chain": None,
        "has_missing_ca": True,
        "has_missing_backbone": True,
        "sha256": None,
        "parse_status": "missing_file",
    }
    if not os.path.exists(path):
        return record

    record["sha256"] = sha256_file(path)

    # Check cache to avoid expensive parsing of files that haven't changed
    cache_key = None
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        cache_key = (path, mtime, size)
        if cache_key in _STRUCTURE_QC_CACHE:
            # Return copy to prevent downstream mutation issues
            cached_record = dict(_STRUCTURE_QC_CACHE[cache_key])
            # Restore the requested sid as the cached record may have had a different
            # sid (or placeholder)
            cached_record["sid"] = sid
            return cached_record
    except Exception:
        pass

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(sid, path)
        assert structure is not None
        model = next(iter(structure))
        chains = list(model.get_chains())
        record["n_chains"] = len(chains)
        if not chains:
            record["parse_status"] = "no_chains"
            return record

        chain = chains[0]  # first-chain selection, matches get_coords_from_pdb
        record["selected_chain"] = chain.id
        residues = list(chain.get_residues())
        coords, valid_mask = get_atom_coordinates(residues, full_backbone=True)

        n_res = int(coords.shape[0])
        n_valid = int(valid_mask.sum())
        record["n_residues"] = n_res
        record["n_valid_residues"] = n_valid
        record["valid_fraction"] = float(n_valid / n_res) if n_res else 0.0

        # Calculate completeness flags over standard (non-HETATM) residues only
        is_standard = np.array([len(r.id[0].strip()) == 0 for r in residues], dtype=bool)
        if len(is_standard) > 0 and is_standard.any():
            record["has_missing_ca"] = bool(np.isnan(coords[is_standard, 0:3]).any())
            record["has_missing_backbone"] = bool(np.isnan(coords[is_standard, 6:12]).any())
        else:
            record["has_missing_ca"] = True
            record["has_missing_backbone"] = True

        record["parse_status"] = "ok" if n_valid > 0 else "no_valid_residues"
    except Exception as exc:
        record["parse_status"] = f"parse_error: {type(exc).__name__}"

    if cache_key is not None:
        _STRUCTURE_QC_CACHE[cache_key] = dict(record)

    return record


def build_structures_table(sids: list[str], pdb_dir: str | Path) -> pd.DataFrame:
    """Build the per-structure QC table for the given sids.

    Args:
        sids: Sorted, de-duplicated structure ids.
        pdb_dir: Directory containing the PDB files (named by sid).

    Returns:
        DataFrame with one row per sid.
    """
    records = [structure_qc(sid, resolve_pdb_path(pdb_dir, sid)) for sid in sids]
    return pd.DataFrame.from_records(records)
