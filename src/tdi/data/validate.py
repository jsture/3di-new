"""Dataset validation: structure QC + CIGAR-semantics checks.

Runs before (or independent of) a build to reject malformed inputs loudly instead
of letting them silently produce bad pairs.
"""

from pathlib import Path
from typing import Any

from tdi.data.cigar import CigarValidationError, validate_cigar
from tdi.data.config import load_config
from tdi.data.pipeline import _read_pairfile
from tdi.data.structures import build_structures_table


def validate_dataset(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """Validate the structures and CIGARs referenced by a dataset config.

    Args:
        config_path: Path to the YAML data config.
        overrides: Optional CLI overrides.

    Returns:
        Summary dict with structure and CIGAR counts.

    Raises:
        CigarValidationError: If any alignment has out-of-range or inconsistent pairs.
    """
    cfg = load_config(config_path, overrides)

    pairfiles = [cfg.dataset.train_pairfile, cfg.dataset.val_pairfile]
    referenced: set[str] = set()
    alignments: list[tuple[int, str, str, str]] = []
    for pf in pairfiles:
        if pf and Path(pf).exists():
            rows = _read_pairfile(pf)
            alignments.extend(rows)
            for _row, sid1, sid2, _cigar in rows:
                referenced.update([sid1, sid2])

    structures = build_structures_table(sorted(referenced), cfg.dataset.pdb_dir)
    n_residues = dict(zip(structures["sid"], structures["n_residues"], strict=True))

    errors: list[str] = []
    n_checked = 0
    for _row, sid1, sid2, cigar in alignments:
        n_ref = int(n_residues.get(sid1, 0))
        n_query = int(n_residues.get(sid2, 0))
        if n_ref == 0 or n_query == 0:
            # Unparseable structure: surfaced by the QC table, skip CIGAR range check.
            continue
        try:
            validate_cigar(cigar, n_ref, n_query)
            n_checked += 1
        except CigarValidationError as exc:
            errors.append(str(exc))

    if errors:
        preview = "\n".join(errors[:10])
        raise CigarValidationError(
            f"{len(errors)} CIGAR validation error(s); first {min(10, len(errors))}:\n{preview}"
        )

    bad_structures = int((structures["parse_status"] != "ok").sum())
    return {
        "n_structures": len(structures),
        "n_structures_not_ok": bad_structures,
        "n_alignments": len(alignments),
        "n_cigars_checked": n_checked,
    }
