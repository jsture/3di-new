"""SCOP classification lookup and fold/superfamily/family decomposition.

The lookup TSV maps ``sid`` to a dotted classification (e.g. ``a.3.1.2``) where
the prefixes give fold (a.3), superfamily (a.3.1), and family (a.3.1.2).
"""

from pathlib import Path


def load_scop_lookup(path: str | Path | None) -> dict[str, str]:
    """Load a SCOP lookup TSV into ``{sid: classification}``.

    Args:
        path: Path to the lookup file, or None.

    Returns:
        Mapping of structure id to classification string; empty if path is None/missing.
    """
    lookup: dict[str, str] = {}
    if path is None:
        return lookup
    lookup_path = Path(path)
    if not lookup_path.exists():
        return lookup
    with open(lookup_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lookup[parts[0]] = parts[1]
    return lookup


def classify(classification: str | None) -> dict[str, str | None]:
    """Split a dotted SCOP classification into fold/superfamily/family levels.

    Args:
        classification: Dotted SCOP string (e.g. ``a.3.1.2``) or None.

    Returns:
        Dict with ``scop``, ``fold``, ``superfamily``, ``family`` (values may be None).
    """
    if not classification:
        return {"scop": None, "fold": None, "superfamily": None, "family": None}
    parts = classification.split(".")
    return {
        "scop": classification,
        "fold": ".".join(parts[:2]) if len(parts) >= 2 else None,
        "superfamily": ".".join(parts[:3]) if len(parts) >= 3 else None,
        "family": classification if len(parts) >= 4 else None,
    }
