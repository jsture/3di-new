"""Learn substitution matrix (submat) from structural alignments of 3Di sequences."""

import argparse
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import util


def load_sequences(seqfile_path: str) -> Dict[str, str]:
    """Load sequences from a space-separated mapping file (sid sequence).

    Args:
        seqfile_path: Path to the sequences mapping file.

    Returns:
        Dictionary mapping structural ID (sid) to 3Di sequence string.
    """
    sid2seq = {}
    with open(seqfile_path, "r") as file:
        for line in file:
            parts = line.rstrip("\n").split()
            if len(parts) >= 2:
                sid, seq = parts[0], parts[1]
                sid2seq[sid] = seq
    return sid2seq


def calc_alphabet_mi(counts: np.ndarray, counts_prev: np.ndarray) -> Tuple[float, float]:
    """Calculate the Mutual Information (MI) and adjusted transition MI.

    Args:
        counts: Joint counts matrix of shape (S, S).
        counts_prev: Lagged joint counts matrix of shape (S, S).

    Returns:
        A tuple of (MI, adjusted transition MI).
    """
    mi = util.mutual_information(counts / counts.sum())
    mi_prev = util.mutual_information(counts_prev / counts_prev.sum())
    mi_tot = mi - (1 - 0.057) * mi_prev  # 0.057 represents baseline transition frequency
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


def write_mat(file_obj, names: List[str], mat: np.ndarray) -> None:
    """Format and write the substitution matrix to a file stream.

    Args:
        file_obj: Open file write stream.
        names: List of state characters.
        mat: Score matrix (shape: (S, S)).
    """
    csize = 4
    header = (" " * (csize - 1)).join([" "] + names)
    file_obj.write(header + "\n")
    for name, line in zip(names, mat):
        file_obj.write(
            "".join([name] + [str(score).rjust(csize, " ") for score in line]) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate 3Di substitution matrix from pair alignments."
    )
    parser.add_argument(
        "pairfile", type=str, help="Path to structural alignment pairfile."
    )
    parser.add_argument(
        "seqfile", type=str, help="Path to csv/txt file mapping sid to 3Di sequence."
    )
    parser.add_argument(
        "--mat", type=str, default=None, help="Output path to save substitution matrix."
    )
    parser.add_argument(
        "--merge_state",
        type=str,
        default="",
        help="Merge an invalid state (e.g. X) into the best matching valid state.",
    )
    parser.add_argument(
        "--cle", action="store_true", help="Force output matrix to use CLE letters."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug matrix info and MI differences."
    )
    args = parser.parse_args()

    # Load sequences
    sid2seq = load_sequences(args.seqfile)

    # Resolve alphabetical characters
    letters = sorted(list({char for seq in sid2seq.values() for char in seq}))
    letter2idx = {letter: k for k, letter in enumerate(letters)}

    # Read pairs and accumulate alignment transition counts
    counts = np.zeros((len(letters), len(letters)), dtype=int)
    counts_prev = np.zeros((len(letters), len(letters)), dtype=int)
    err_cnt = 0

    with open(args.pairfile, "r") as pair_file:
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

                # Check if prior residue is also aligned
                if j > 0 and idx_2[k - 1] == j - 1:
                    row = letter2idx[seq1[i]]
                    col = letter2idx[seq2[j - 1]]
                    counts_prev[row, col] += 1
                if i > 0 and idx_1[k - 1] == i - 1:
                    row = letter2idx[seq2[j]]
                    col = letter2idx[seq1[i - 1]]
                    counts_prev[row, col] += 1

    # Merge an invalid/low-frequency state dynamically into a valid state
    if args.merge_state:
        idx_invalid_state = "".join(letters).find(args.merge_state)
        assert (
            idx_invalid_state >= 0
        ), f"State '{args.merge_state}' is not found in the alphabet"

        _, mi_tot_no_merge = calc_alphabet_mi(counts, counts_prev)
        mi_opt, idx_opt = -1.0, 0

        for j in range(len(counts)):
            if j == idx_invalid_state:
                continue
            new_counts = merge_columns(counts, idx_invalid_state, j)
            new_counts_prev = merge_columns(counts_prev, idx_invalid_state, j)
            _, mi_tot = calc_alphabet_mi(new_counts, new_counts_prev)

            if args.verbose:
                print(
                    f"Merging {args.merge_state} into {letters[j]} cost "
                    f"{(mi_tot_no_merge - mi_tot):.4f} bit of MI.",
                    file=sys.stderr,
                )
            if mi_tot > mi_opt:
                mi_opt, idx_opt = mi_tot, j

        if not args.verbose:
            _, mi_tot = calc_alphabet_mi(counts, counts_prev)
            print(
                f"Merging {args.merge_state} cost {(mi_tot - mi_opt):.4f} bit of MI.",
                file=sys.stderr,
            )

        counts = merge_columns(counts, idx_invalid_state, idx_opt)
        counts_prev = merge_columns(counts_prev, idx_invalid_state, idx_opt)
        print(f"assign_invalid_states_to = {letters[idx_opt]}")
        letters.pop(idx_invalid_state)

    if args.verbose:
        np.savetxt(sys.stdout, counts.astype(int), fmt="%d")

    # Calculate log odds scores (half-bits)
    p_ab = counts / counts.sum()
    p_a = p_ab.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = 2 * np.log2(p_ab / (p_a * p_a[:, np.newaxis]))
    scores[~np.isfinite(scores)] = 0
    scores = np.rint(scores).astype(int)

    # Output substitution matrix file
    if args.mat:
        out_letters = letters
        if args.cle:
            out_letters = list("ACDEFGHIKLMNPQRSTVWYX")[: len(letters)]
        with open(args.mat, "w") as file:
            write_mat(file, out_letters, scores)

    # Print summary and dataframe representation
    df = pd.DataFrame(scores, index=letters, columns=letters)
    print(df)

    mi, mi_tot = calc_alphabet_mi(counts, counts_prev)
    print(f"MI = {mi:.4f}")
    print(f"MI_tot = {mi_tot:.4f}")
    print(f"counts = {counts.sum()}")


if __name__ == "__main__":
    main()
