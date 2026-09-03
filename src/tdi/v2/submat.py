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


def mutual_information_from_counts(counts: np.ndarray) -> float:
    """Plug-in mutual information, in bits, of a joint count matrix.

    The single place counts become an MI number, shared by final evaluation, the training-time
    ``val_pair_mi`` diagnostic, and the descriptor-information probe, so all three report the
    same quantity computed the same way.

    Args:
        counts: Joint count matrix of shape (A, B). Need not be square or symmetric.

    Returns:
        Mutual information in bits; 0.0 for an empty matrix.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    return util.mutual_information(counts / total)


def miller_madow_corrected_mi(counts: np.ndarray, n_observations: int | None = None) -> float:
    """Apply the Miller-Madow bias correction to a plug-in MI estimate.

    The plug-in estimator is biased at finite sample size by roughly
    ``(m_xy - m_x - m_y + 1) / (2 N)`` nats, where each ``m`` is the number of *occupied*
    bins in the corresponding support. Subtracting that term is what this function does.

    The sign is not fixed. It is negative -- lowering an inflated estimate -- only when the
    joint occupies more bins than the two marginals combined, which is the usual sparse
    many-bin case. For a table whose occupancy is concentrated (a near-diagonal joint, where
    ``m_xy`` approaches ``m_x`` and ``m_y``) the correction is positive and can push the
    result above ``log2(K)``. Callers must not assume it always reduces the estimate.

    **Validity.** This is the ordinary multinomial correction: it assumes ``counts`` came
    from ``N`` independent draws from the joint distribution. It is therefore *not* valid on
    a symmetrized table, where every observation was written into two dependent cells --
    passing the one-orientation ``N`` does not repair the independence assumption or the
    degrees of freedom. Apply it to the unsymmetrized joint, and use a pair-level resampling
    estimate for the bias of a symmetrized target.

    Args:
        counts: Joint count matrix of shape (A, B), from independent draws.
        n_observations: Independent observations behind ``counts``. Defaults to the matrix
            total, which is correct whenever each observation contributed exactly one count.

    Returns:
        Bias-corrected mutual information in bits.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    n = int(n_observations) if n_observations is not None else int(total)
    if n <= 0:
        return 0.0

    occupied_joint = int(np.count_nonzero(counts))
    occupied_x = int(np.count_nonzero(counts.sum(axis=1)))
    occupied_y = int(np.count_nonzero(counts.sum(axis=0)))
    correction = (occupied_x + occupied_y - occupied_joint - 1) / (2 * n * np.log(2))
    return mutual_information_from_counts(counts) + float(correction)


def calc_alphabet_mi(counts: np.ndarray, counts_prev: np.ndarray) -> tuple[float, float]:
    """Calculate the Mutual Information (MI) and adjusted transition MI.

    Args:
        counts: Joint counts matrix of shape (S, S).
        counts_prev: Lagged joint counts matrix of shape (S, S).

    Returns:
        A tuple of (MI, adjusted transition MI).
    """
    mi = mutual_information_from_counts(counts)
    # Guard the lagged matrix: no adjacent pairs -> zero baseline (avoid 0/0).
    mi_prev = mutual_information_from_counts(counts_prev)
    # Adjust for sequential dependency baseline
    mi_tot = mi - (1 - 0.057) * mi_prev
    return mi, mi_tot


def chain_contribution(mi: float, mi_tot: float) -> tuple[float, float]:
    """Recover the lagged MI term and its share of raw MI from the reported pair.

    ``mi_tot = mi - (1 - 0.057) * mi_prev``, so ``mi_prev`` is recoverable from the two
    reported numbers. ``chain_fraction`` is the share of raw MI that the transition
    adjustment removes -- the diagnostic that separates an alphabet carrying genuine
    alignment signal from one inflating raw MI with longer correlated state runs.

    ``mi_prev`` is recoverable whenever both numbers are known, including when ``mi`` is 0:
    a zero raw MI with positive lagged MI gives a negative ``mi_tot``, and the lagged term is
    still exactly ``-mi_tot / (1 - 0.057)``. Only the *fraction* is undefined there, since it
    divides by ``mi``, so the guard covers the division alone.

    Args:
        mi: Raw pair mutual information in bits.
        mi_tot: Transition-adjusted mutual information in bits.

    Returns:
        A tuple of (mi_prev, chain_fraction). ``chain_fraction`` is 0.0 when ``mi`` is 0.
    """
    mi_prev = (mi - mi_tot) / (1 - 0.057)
    if mi == 0.0:
        return mi_prev, 0.0
    return mi_prev, (mi - mi_tot) / mi


def merge_columns(counts: np.ndarray, i: int, j: int) -> np.ndarray:
    """Merge row and column index i into index j in a square matrix.

    Args:
        counts: Original square matrix of shape (S, S).
        i: Index to merge from (deleted index).
        j: Index to merge into (retained index).

    Returns:
        Reduced matrix of shape (S-1, S-1).
    """
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"counts must be square, got shape {counts.shape}")
    if not (0 <= i < len(counts)) or not (0 <= j < len(counts)):
        raise IndexError(f"merge indices {(i, j)} are out of range for size {len(counts)}")
    if i == j:
        raise ValueError("Cannot merge a state into itself.")

    mask = np.ones(len(counts), dtype=bool)
    mask[i] = False
    retained_j = j - 1 if i < j else j

    new_counts = np.copy(counts[mask, :][:, mask])
    new_counts[retained_j, :] += counts[i, mask]
    new_counts[:, retained_j] += counts[mask, i]
    new_counts[retained_j, retained_j] += counts[i, i]

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
            res = util.parse_pairfile_line(line)
            if res is None:
                continue
            sid1, sid2, cigar_string = res
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
                # Bounds check to prevent out-of-bounds indexing on seq1/seq2
                if i >= len(seq1) or j >= len(seq2):
                    continue
                # Skip positions whose state is not in the alphabet (e.g. the invalid
                # state for residues without valid descriptors): they have no index.
                a1 = letter2idx.get(seq1[i])
                a2 = letter2idx.get(seq2[j])
                if a1 is None or a2 is None:
                    continue
                counts[a1, a2] += 1
                counts[a2, a1] += 1

                # Lagged counts accumulation for transition adjustments.
                # Restrict updates to k > 0 to prevent a negative index lag check.
                if k > 0 and j > 0 and idx_2[k - 1] == j - 1:
                    prev = letter2idx.get(seq2[j - 1])
                    if prev is not None:
                        counts_prev[a1, prev] += 1
                if k > 0 and i > 0 and idx_1[k - 1] == i - 1:
                    prev = letter2idx.get(seq1[i - 1])
                    if prev is not None:
                        counts_prev[a2, prev] += 1

    return counts, counts_prev
