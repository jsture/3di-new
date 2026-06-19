#!/usr/bin/env python3
"""CLI: split SCOP domain identifiers into cross-validation fold partitions."""

import argparse
import random
from pathlib import Path


def get_fold(classification: str) -> str:
    """Extract class and fold identifier from a full SCOP classification string.

    For example, maps "d.58.1.5" to "d.58".

    Args:
        classification: SCOP classification string.

    Returns:
        The class + fold portion of the identifier.
    """
    return ".".join(classification.split(".")[:2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split SCOP classification targets into K random validation folds."
    )
    parser.add_argument(
        "--lookup-file",
        type=str,
        default="data/scop_lookup.tsv",
        help="Path to the SCOP classification lookup mapping TSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="tmp",
        help="Output directory to write partitioned fold lists.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of validation folds to construct.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of folds shuffling.",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.lookup_file) as file:
        lines = file.readlines()

    sids: list[str] = []
    cls: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            sids.append(parts[0])
            cls.append(parts[1])

    folds = sorted({get_fold(c) for c in cls})

    random.seed(args.seed)
    random.shuffle(folds)

    n = len(folds)
    chunk_sizes = [n // args.k] * args.k
    for i in range(n - sum(chunk_sizes)):
        chunk_sizes[i] += 1

    splits: list[list[str]] = []
    for i, size in enumerate(chunk_sizes):
        start_idx = sum(chunk_sizes[:i])
        splits.append(folds[start_idx : start_idx + size])

    for i, split in enumerate(splits):
        split_set = set(split)
        fold_lines = [f"{sid} {cl}" for sid, cl in zip(sids, cls) if get_fold(cl) in split_set]
        split_file = out_path / f"fold_split{i}.txt"
        with open(split_file, "w") as file:
            file.write("\n".join(fold_lines))


if __name__ == "__main__":
    main()
