"""Content-hashing helpers for the data pipeline manifest.

Provides sha256 over files and numpy arrays plus the current git commit, so the
manifest can prove that identical inputs+config yield record-identical outputs.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Return the hex sha256 of a numpy array's dtype, shape, and raw bytes.

    Uses a C-contiguous copy so the digest is independent of memory layout.
    """
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def array_record(arr: np.ndarray) -> dict[str, object]:
    """Return a manifest record (sha256, shape, dtype, n_rows) for an array."""
    return {
        "sha256": sha256_array(arr),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_rows": int(arr.shape[0]) if arr.ndim else 0,
    }


def git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
