#!/usr/bin/env python3
"""CLI: extract features from aligned structure pairs to compile VQ-VAE training data."""

import argparse
import random
from pathlib import Path

import numpy as np

from tdi.training_data import align_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from aligned structure pairs to compile VQ-VAE training data."
    )
    parser.add_argument("pdb_dir", type=str, help="Directory containing PDB files.")
    parser.add_argument("pairfile", type=str, help="Path to structural alignment pairfile.")
    parser.add_argument("alpha", type=float, help="Virtual center offset angle alpha.")
    parser.add_argument("beta", type=float, help="Virtual center offset angle beta.")
    parser.add_argument("d", type=float, help="Virtual center offset distance d.")
    parser.add_argument("out", type=str, help="Output path for the .npy training data file.")
    args = parser.parse_args()

    pdbs_train_path = Path(__file__).parent.parent / "data" / "pdbs_train.txt"

    with open(pdbs_train_path) as file:
        pdbs_train = set(file.read().splitlines())

    alignments: list[tuple[str, str, str]] = []
    with open(args.pairfile) as file:
        for line in file:
            parts = line.rstrip("\n").split()
            if len(parts) >= 3:
                sid1, sid2, cigar_string = parts[0], parts[1], parts[2]
                if sid1 in pdbs_train and sid2 in pdbs_train:
                    alignments.append((sid1, sid2, cigar_string))

    random.Random(123).shuffle(alignments)

    virtual_center = (args.alpha, args.beta, args.d)

    xy: list[tuple[np.ndarray, np.ndarray]] = []
    for sid1, sid2, cigar_string in alignments:
        xy.append(align_features(args.pdb_dir, virtual_center, sid1, sid2, cigar_string))

    if xy:
        x_feat = np.vstack([x for x, y in xy])
        y_feat = np.vstack([y for x, y in xy])
        idx = np.arange(len(x_feat))
        np.random.RandomState(123).shuffle(idx)
        np.save(args.out, np.dstack([x_feat[idx], y_feat[idx]]))
    else:
        print("Warning: No alignments matched training PDB set. Output not written.")


if __name__ == "__main__":
    main()
