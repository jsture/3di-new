#!/usr/bin/env python3
"""CLI: extract features from aligned structure pairs to compile VQ-VAE training data."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from tdi.v2.training_data import align_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from aligned structure pairs to compile VQ-VAE training data."
    )
    parser.add_argument("pdb_dir", type=str, help="Directory containing PDB files.")
    parser.add_argument("pairfile", type=str, help="Path to structural alignment pairfile.")
    parser.add_argument("alpha", type=float, help="Virtual center offset angle alpha.")
    parser.add_argument("beta", type=float, help="Virtual center offset angle beta.")
    parser.add_argument("d", type=float, help="Virtual center offset distance d.")
    parser.add_argument("out_dir", type=str, help="Output directory for training data artifacts.")
    parser.add_argument(
        "--manifest", type=str, help="Path to split manifest CSV (e.g. train_manifest.csv)."
    )
    parser.add_argument(
        "--max_pairs", type=int, default=512, help="Max residue pairs per alignment."
    )
    parser.add_argument(
        "--max_ca_dist",
        type=float,
        default=5.0,
        help="Max C-alpha distance filter after superposition.",
    )
    args = parser.parse_args()

    # Load allowed SIDs from manifest
    allowed_sids = set()
    if args.manifest:
        df_manifest = pd.read_csv(args.manifest)
        allowed_sids = set(df_manifest["structure_id"].values)
    else:
        pdbs_train_path = Path(__file__).parent.parent / "data" / "raw" / "pdbs_train.txt"
        if pdbs_train_path.exists():
            with open(pdbs_train_path) as file:
                allowed_sids = set(file.read().splitlines())
        else:
            print("Warning: No manifest provided and pdbs_train.txt not found. Using all SIDs.")

    alignments: list[tuple[str, str, str]] = []
    with open(args.pairfile) as file:
        for line in file:
            parts = line.rstrip("\n").split()
            if len(parts) >= 3:
                sid1, sid2, cigar_string = parts[0], parts[1], parts[2]
                if not allowed_sids or (sid1 in allowed_sids and sid2 in allowed_sids):
                    alignments.append((sid1, sid2, cigar_string))

    random.Random(123).shuffle(alignments)

    virtual_center = (args.alpha, args.beta, args.d)

    xy_list = []
    meta_list = []

    for sid1, sid2, cigar_string in alignments:
        try:
            x, y, meta = align_features(
                args.pdb_dir, virtual_center, sid1, sid2, cigar_string, max_ca_dist=args.max_ca_dist
            )

            if len(x) > 0:
                if len(x) > args.max_pairs:
                    # Randomly sub-sample
                    idx = np.random.choice(len(x), args.max_pairs, replace=False)
                    x = x[idx]
                    y = y[idx]
                    for k in meta:
                        if meta[k] is not None:
                            meta[k] = np.array(meta[k])[idx]

                xy_list.append((x, y))

                # Expand meta dict to rows
                for i in range(len(x)):
                    row = {
                        "sid_source": meta["sid_source"][i],
                        "sid_target": meta["sid_target"][i],
                        "idx_source": meta["idx_source"][i],
                        "idx_target": meta["idx_target"][i],
                        "ca_dist": meta["ca_dist"][i] if meta["ca_dist"] is not None else None,
                    }
                    meta_list.append(row)
        except Exception as e:
            print(f"Error processing {sid1}-{sid2}: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if xy_list:
        x_feat = np.vstack([x for x, y in xy_list])
        y_feat = np.vstack([y for x, y in xy_list])

        # We save x and y explicitly instead of dstack for clarity, but maintaining legacy behavior
        # Since v2 train script uses training_data_raw[:, :, 0], we stick to dstack.
        stacked = np.dstack([x_feat, y_feat])
        np.save(out_dir / "data.npy", stacked)

        # Save metadata
        df_meta = pd.DataFrame(meta_list)
        df_meta.to_parquet(out_dir / "metadata.parquet", index=False)

        print(f"Generated {len(x_feat)} pairs from {len(xy_list)} alignments.")
        print(f"Saved artifacts to {out_dir}")
    else:
        print("Warning: No alignments produced valid pairs. Output not written.")


if __name__ == "__main__":
    main()
