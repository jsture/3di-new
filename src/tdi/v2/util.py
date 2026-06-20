"""Utility helper functions for parsing CIGAR strings and calculating mutual information for v2.

This module contains alignment parsing for CIGAR formatted alignment logs and Shannon entropy-based
mutual information scoring for substitution statistics.
"""

import re

import numpy as np


def parse_cigar(cigar_string: str) -> np.ndarray:
    """Parse a CIGAR alignment string into a coordinate index mapping of matching positions.

    Supported actions:
        - D (deletion in query): advances reference index.
        - I (insertion in query): advances query index.
        - M (match/mismatch): advances both reference and query index (not returned).
        - P (perfect match): advances both, and records mapped alignment pair indices.

    Args:
        cigar_string: CIGAR alignment string (e.g. "10M5D3P").

    Returns:
        NumPy array of shape (N, 2) containing aligned index pairs (index_ref, index_query).
    """
    ref, query = 0, 0
    matches = []

    # Match counts followed by any letter op so unsupported ops are rejected (not
    # silently skipped, which would desynchronize the reference/query indices).
    for cnt_str, action in re.findall(r"([0-9]*)([A-Za-z])", cigar_string):
        cnt = int(cnt_str) if cnt_str else 1

        if action == "D":
            ref += cnt
        elif action == "I":
            query += cnt
        elif action == "M":
            ref += cnt
            query += cnt
        elif action == "P":
            matches.extend([(ref + i, query + i) for i in range(cnt)])
            ref += cnt
            query += cnt
        else:
            raise ValueError(f"Action {action} is not supported.")

    return np.array(matches, dtype=np.int64).reshape(-1, 2)


def mutual_information(p_ab: np.ndarray) -> float:
    """Calculate the mutual information of a joint probability distribution p(a, b).

    Args:
        p_ab: Joint probability matrix of shape (S, S).

    Returns:
        Mutual information in bits.
    """
    p_a = p_ab.sum(axis=1)
    p_b = p_ab.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Log2 scores calculation for MI
        log_scores = np.log2(p_ab / (p_a[:, np.newaxis] * p_b))
        return float(np.sum(p_ab * log_scores, where=np.isfinite(log_scores)))
