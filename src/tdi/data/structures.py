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
        record["has_missing_ca"] = bool(np.isnan(coords[:, 0:3]).any())
        record["has_missing_backbone"] = bool(np.isnan(coords[:, 6:12]).any())
        record["parse_status"] = "ok" if n_valid > 0 else "no_valid_residues"
    except Exception as exc:
        record["parse_status"] = f"parse_error: {type(exc).__name__}"

    return record


def build_structures_table(sids: list[str], pdb_dir: str | Path) -> pd.DataFrame:
    """Build the per-structure QC table for the given sids.

    Args:
        sids: Sorted, de-duplicated structure ids.
        pdb_dir: Directory containing the PDB files (named by sid).

    Returns:
        DataFrame with one row per sid.
    """
    records = [structure_qc(sid, os.path.join(str(pdb_dir), sid)) for sid in sids]
    return pd.DataFrame.from_records(records)
