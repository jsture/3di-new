#!/usr/bin/env python3
"""CLI: split SCOP domain identifiers into cross-validation fold partitions."""

import argparse
import random
from pathlib import Path

from tdi.data.scop import classify, load_scop_lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split SCOP classification targets into K random validation folds."
    )
    parser.add_argument(
        "--lookup-file",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "raw" / "scop_lookup.tsv"),
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

    scop_lookup = load_scop_lookup(args.lookup_file)
    sids: list[str] = list(scop_lookup.keys())
    cls: list[str] = list(scop_lookup.values())

    folds = sorted({fold for c in cls if (fold := classify(c)["fold"]) is not None})

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
        fold_lines = [
            f"{sid} {cl}" for sid, cl in zip(sids, cls) if classify(cl)["fold"] in split_set
        ]
        split_file = out_path / f"fold_split{i}.txt"
        with open(split_file, "w") as file:
            file.write("\n".join(fold_lines))


if __name__ == "__main__":
    main()
