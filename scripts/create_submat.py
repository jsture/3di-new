#!/usr/bin/env python3
"""CLI: learn substitution matrix (submat) from structural alignments of 3Di sequences."""

import argparse
import sys

import numpy as np
import pandas as pd

from tdi.submat import (
    accumulate_counts,
    calc_alphabet_mi,
    load_sequences,
    merge_columns,
    write_mat,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate 3Di substitution matrix from pair alignments."
    )
    parser.add_argument("pairfile", type=str, help="Path to structural alignment pairfile.")
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

    sid2seq = load_sequences(args.seqfile)

    letters = sorted({char for seq in sid2seq.values() for char in seq})
    letter2idx = {letter: k for k, letter in enumerate(letters)}

    counts, counts_prev = accumulate_counts(args.pairfile, sid2seq, letter2idx, len(letters))

    if args.merge_state:
        idx_invalid_state = "".join(letters).find(args.merge_state)
        assert idx_invalid_state >= 0, f"State '{args.merge_state}' is not found in the alphabet"

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

    p_ab = counts / counts.sum()
    p_a = p_ab.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = 2 * np.log2(p_ab / (p_a * p_a[:, np.newaxis]))
    scores[~np.isfinite(scores)] = 0
    scores = np.rint(scores).astype(int)

    if args.mat:
        out_letters = letters
        if args.cle:
            out_letters = list("ACDEFGHIKLMNPQRSTVWYX")[: len(letters)]
        with open(args.mat, "w") as file:
            write_mat(file, out_letters, scores)

    df = pd.DataFrame(scores, index=letters, columns=letters)
    print(df)

    mi, mi_tot = calc_alphabet_mi(counts, counts_prev)
    print(f"MI = {mi:.4f}")
    print(f"MI_tot = {mi_tot:.4f}")
    print(f"counts = {counts.sum()}")


if __name__ == "__main__":
    main()
