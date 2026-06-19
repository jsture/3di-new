"""Substitution matrix construction from structural alignments of 3Di sequences for v2.

This module constructs scoring substitution matrices from sequences aligned structurally,
evaluating state transitions and calculating mutual information.
"""

import sys

import numpy as np

from . import util


def load_sequences(seqfile_path: str) -> dict[str, str]:
    """Load sequences from a space-separated mapping file (sid sequence).

    Args:
        seqfile_path: Path to the sequences mapping file.

    Returns:
        Dictionary mapping structural ID (sid) to 3Di sequence string.
    """
    sid2seq = {}
    with open(seqfile_path) as file:
        for line in file:
            parts = line.rstrip("\n").split()
            if len(parts) >= 2:
                sid, seq = parts[0], parts[1]
                sid2seq[sid] = seq
    return sid2seq


def calc_alphabet_mi(counts: np.ndarray, counts_prev: np.ndarray) -> tuple[float, float]:
    """Calculate the Mutual Information (MI) and adjusted transition MI.

    Args:
        counts: Joint counts matrix of shape (S, S).
        counts_prev: Lagged joint counts matrix of shape (S, S).

    Returns:
        A tuple of (MI, adjusted transition MI).
    """
    mi = util.mutual_information(counts / counts.sum())
    mi_prev = util.mutual_information(counts_prev / counts_prev.sum())
    # Adjust for sequential dependency baseline
    mi_tot = mi - (1 - 0.057) * mi_prev
    return mi, mi_tot


def merge_columns(counts: np.ndarray, i: int, j: int) -> np.ndarray:
    """Merge row and column index i into index j in a square matrix.

    Args:
        counts: Original square matrix of shape (S, S).
        i: Index to merge from (deleted index).
        j: Index to merge into (retained index).

    Returns:
        Reduced matrix of shape (S-1, S-1).
    """
    mask = np.ones(len(counts), dtype=bool)
    mask[i] = False

    new_counts = np.copy(counts[mask, :][:, mask])
    new_counts[j, :] += counts[i, mask]
    new_counts[:, j] += counts[mask, i]
    new_counts[j, j] += counts[i, i]

    return new_counts


def write_mat(file_obj, names: list[str], mat: np.ndarray) -> None:
    """Format and write the substitution matrix to a file stream.

    Args:
        file_obj: Open file write stream.
        names: List of state characters.
        mat: Score matrix (shape: (S, S)).
    """
    csize = 4
    header = (" " * (csize - 1)).join([" ", *names])
    file_obj.write(header + "\n")
    for name, line in zip(names, mat):
        file_obj.write("".join([name] + [str(score).rjust(csize, " ") for score in line]) + "\n")


def accumulate_counts(
    pairfile_path: str,
    sid2seq: dict[str, str],
    letter2idx: dict[str, int],
    n_letters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read pair alignments and accumulate state transition counts.

    Args:
        pairfile_path: Path to structural alignment pairfile.
        sid2seq: Mapping of structural ID to 3Di sequence.
        letter2idx: Mapping of alphabet character to index.
        n_letters: Size of the alphabet.

    Returns:
        A tuple of (counts, counts_prev) joint count matrices.
    """
    counts = np.zeros((n_letters, n_letters), dtype=int)
    counts_prev = np.zeros((n_letters, n_letters), dtype=int)
    err_cnt = 0

    with open(pairfile_path) as pair_file:
        for line in pair_file:
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            sid1, sid2, cigar_string = parts[0], parts[1], parts[2]
            seq1 = sid2seq.get(sid1)
            seq2 = sid2seq.get(sid2)

            if not seq1 or not seq2:
                if err_cnt < 100:
                    missing_sid = sid1 if not seq1 else sid2
                    print(f"Not found: {missing_sid}", file=sys.stderr)
                    err_cnt += 1
                elif err_cnt == 100:
                    print("Errors truncated...", file=sys.stderr)
                    err_cnt += 1
                continue

            idx_pairs = util.parse_cigar(cigar_string)
            if idx_pairs.size == 0:
                continue

            idx_1, idx_2 = idx_pairs.T
            for k in range(idx_1.shape[0]):
                i, j = idx_1[k], idx_2[k]
                row = letter2idx[seq1[i]]
                col = letter2idx[seq2[j]]
                counts[row, col] += 1
                counts[col, row] += 1

                # Lagged counts accumulation for transition adjustments
                if j > 0 and idx_2[k - 1] == j - 1:
                    row = letter2idx[seq1[i]]
                    col = letter2idx[seq2[j - 1]]
                    counts_prev[row, col] += 1
                if i > 0 and idx_1[k - 1] == i - 1:
                    row = letter2idx[seq2[j]]
                    col = letter2idx[seq1[i - 1]]
                    counts_prev[row, col] += 1

    return counts, counts_prev
