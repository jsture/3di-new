#!/usr/bin/env python3
"""CLI: Generate group-aware train and validation split manifests."""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate group-aware train and validation split manifests."
    )
    parser.add_argument("input", type=str, help="Path to input PDBs list (e.g., pdbs_train.txt).")
    parser.add_argument("out_dir", type=str, help="Output directory for manifests.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--group_by",
        type=str,
        default="superfamily",
        choices=["fold", "superfamily", "pdb"],
        help="Grouping level to partition splits.",
    )
    parser.add_argument(
        "--scop_lookup",
        type=str,
        default="data/raw/scop_lookup.tsv",
        help="Path to SCOP classification lookup mapping TSV file.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        sids = f.read().splitlines()

    scop_lookup = {}
    if args.group_by in ("fold", "superfamily"):
        lookup_path = Path(args.scop_lookup)
        if lookup_path.exists():
            with open(lookup_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        scop_lookup[parts[0]] = parts[1]
        else:
            print(
                f"Warning: SCOP lookup file {args.scop_lookup} not found. "
                "Falling back to PDB grouping."
            )

    # Group by PDB ID to prevent cross-domain leakage
    # SCOP SID format is typically d[pdb_id][chain][domain] -> e.g., d1qksa1
    # PDB ID is usually chars 1 to 5 (4 chars long). If not standard, group by the whole SID.
    groups: dict[str, list[str]] = defaultdict(list)
    fallback_count = 0
    for sid in sids:
        group_id = None
        if args.group_by in ("fold", "superfamily") and sid in scop_lookup:
            classification = scop_lookup[sid]
            parts = classification.split(".")
            if args.group_by == "fold":
                group_id = ".".join(parts[:2])
            else:
                group_id = ".".join(parts[:3])

        # Fallback to PDB grouping if SCOP class lookup fails
        if group_id is None:
            if args.group_by in ("fold", "superfamily") and len(scop_lookup) > 0:
                fallback_count += 1
            if sid.startswith("d") and len(sid) >= 5:
                group_id = sid[1:5]
            else:
                group_id = sid
        groups[group_id].append(sid)

    # Shuffle groups
    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    # Split
    val_size = int(len(group_ids) * args.val_split)
    val_groups = set(group_ids[:val_size])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (
        open(out_dir / "train_manifest.csv", "w") as f_train,
        open(out_dir / "val_manifest.csv", "w") as f_val,
    ):
        f_train.write("structure_id,group_id,split\n")
        f_val.write("structure_id,group_id,split\n")

        train_count = 0
        val_count = 0
        for group_id, group_sids in groups.items():
            is_val = group_id in val_groups
            for sid in group_sids:
                if is_val:
                    f_val.write(f"{sid},{group_id},val\n")
                    val_count += 1
                else:
                    f_train.write(f"{sid},{group_id},train\n")
                    train_count += 1

    if fallback_count > 0:
        print(f"Homology grouping: {fallback_count} SIDs fell back to PDB ID grouping.")
    print(f"Total unique groups: {len(groups)}")
    print(f"Split complete: {train_count} train SIDs, {val_count} val SIDs.")
    print(f"Manifests saved to {out_dir}")


if __name__ == "__main__":
    main()
