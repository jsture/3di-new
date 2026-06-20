"""CIGAR-semantics validation for the custom P=aligned-pair convention.

Fails loudly on malformed or foreign-tool CIGARs rather than silently producing
out-of-range index pairs.
"""

import re

import numpy as np

from tdi.v2.util import parse_cigar

_CIGAR_RE = re.compile(r"(?:\d*[IDMP])+")
_TOKEN_RE = re.compile(r"([0-9]*)([IDMP])")


class CigarValidationError(ValueError):
    """Raised when a CIGAR string violates the expected pair semantics."""


def validate_cigar(cigar_string: str, n_ref: int, n_query: int) -> np.ndarray:
    """Parse and validate a CIGAR string against structure lengths.

    Checks that aligned index pairs are non-negative, strictly within both
    structures' residue ranges, and that the parsed pair count matches the number
    of ``P`` (aligned-pair) positions implied by the CIGAR.

    Args:
        cigar_string: CIGAR alignment string using the ``P``=aligned-pair convention.
        n_ref: Number of residues in the reference (source) structure.
        n_query: Number of residues in the query (target) structure.

    Returns:
        Parsed index pairs of shape (N, 2).

    Raises:
        CigarValidationError: On any out-of-range, negative, or count-inconsistent pair.

    Note:
        Indices are assumed 0-based into the same residue enumeration that produced
        ``n_ref``/``n_query`` — i.e. ``list(chain.get_residues())`` of the first chain,
        HETATM rows included (see ``get_atom_coordinates``). If the aligner that emitted the
        CIGAR numbered residues differently, in-range-but-misaligned pairs pass this check.
    """
    if not cigar_string or _CIGAR_RE.fullmatch(cigar_string) is None:
        raise CigarValidationError(f"CIGAR {cigar_string!r} is malformed.")

    tokens: list[tuple[int, str]] = []
    for cnt_str, action in _TOKEN_RE.findall(cigar_string):
        cnt = int(cnt_str) if cnt_str else 1
        if cnt <= 0:
            raise CigarValidationError(
                f"CIGAR {cigar_string!r}: zero-length operation {cnt}{action}."
            )
        tokens.append((cnt, action))

    ref = 0
    query = 0
    for cnt, action in tokens:
        if action == "D":
            ref += cnt
        elif action == "I":
            query += cnt
        elif action in ("M", "P"):
            ref += cnt
            query += cnt
        if ref > n_ref:
            raise CigarValidationError(f"CIGAR {cigar_string!r}: ref cursor {ref} > n_ref {n_ref}.")
        if query > n_query:
            raise CigarValidationError(
                f"CIGAR {cigar_string!r}: query cursor {query} > n_query {n_query}."
            )

    try:
        pairs = parse_cigar(cigar_string)
    except ValueError as exc:
        raise CigarValidationError(f"CIGAR {cigar_string!r} is malformed: {exc}") from exc

    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise CigarValidationError(f"CIGAR {cigar_string!r} did not parse to (N, 2) pairs.")

    # Expected pair count: sum of P-run lengths (P defaults to count 1 when bare).
    expected = sum(cnt for cnt, action in tokens if action == "P")
    if pairs.shape[0] != expected:
        raise CigarValidationError(
            f"CIGAR {cigar_string!r}: parsed {pairs.shape[0]} pairs but expected {expected} "
            "from P-run lengths."
        )

    if pairs.min() < 0:
        raise CigarValidationError(f"CIGAR {cigar_string!r}: negative index pair.")

    ref_idx, query_idx = pairs[:, 0], pairs[:, 1]
    if ref_idx.max() >= n_ref:
        raise CigarValidationError(
            f"CIGAR {cigar_string!r}: ref index {int(ref_idx.max())} >= n_ref {n_ref}."
        )
    if query_idx.max() >= n_query:
        raise CigarValidationError(
            f"CIGAR {cigar_string!r}: query index {int(query_idx.max())} >= n_query {n_query}."
        )

    return pairs
