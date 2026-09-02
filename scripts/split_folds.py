#!/usr/bin/env python3
"""CLI: split SCOP domain identifiers into cross-validation fold partitions."""

import argparse
from pathlib import Path

from sklearn.model_selection import KFold

from tdi.data.scop import classify, load_scop_lookup


def partition_folds(folds: list[str], k: int, seed: int) -> list[list[str]]:
    """Partition unique SCOP folds into ``k`` seeded validation sets."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if len(folds) < k:
        raise ValueError(f"k={k} exceeds the number of classified folds ({len(folds)})")

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        [folds[int(index)] for index in test_indices] for _, test_indices in splitter.split(folds)
    ]


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

    try:
        splits = partition_folds(folds, args.k, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

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
