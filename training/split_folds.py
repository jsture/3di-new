"""Split SCOP domain identifiers into cross-validation fold partitions."""

import argparse
from pathlib import Path
import random
from typing import List


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

    # Create output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load lookup mapping
    with open(args.lookup_file, "r") as file:
        lines = file.readlines()

    sids: List[str] = []
    cls: List[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            sids.append(parts[0])
            cls.append(parts[1])

    # Distinct folds
    folds = sorted(list({get_fold(c) for c in cls}))

    # Shuffle folds with reproducibility
    random.seed(args.seed)
    random.shuffle(folds)

    # Distribute fold IDs into K splits
    n = len(folds)
    chunk_sizes = [n // args.k] * args.k
    for i in range(n - sum(chunk_sizes)):
        chunk_sizes[i] += 1

    splits: List[List[str]] = []
    for i, size in enumerate(chunk_sizes):
        start_idx = sum(chunk_sizes[:i])
        splits.append(folds[start_idx : start_idx + size])

    # Write split lists to files
    for i, split in enumerate(splits):
        fold_lines: List[str] = []
        for sid, cl in zip(sids, cls):
            if get_fold(cl) in split:
                fold_lines.append(f"{sid} {cl}")

        split_file = out_path / f"fold_split{i}.txt"
        with open(split_file, "w") as file:
            file.write("\n".join(fold_lines))


if __name__ == "__main__":
    main()
