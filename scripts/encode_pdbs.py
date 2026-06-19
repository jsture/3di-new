#!/usr/bin/env python3
"""CLI: convert PDB structure coordinates to discrete 3Di state sequences."""

import argparse
import sys

import numpy as np
import torch

from tdi.v1.encode import process_pdb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB structure coordinates to discrete 3Di state sequences."
    )
    parser.add_argument("encoder", type=str, help="Path to trained *.pt encoder model.")
    parser.add_argument("centroids", type=str, help="Path to states.txt centroids file.")
    parser.add_argument(
        "--pdb_dir", type=str, required=True, help="Directory containing PDB files."
    )
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        required=True,
        help="Virtual center parameters (alpha, beta, d).",
    )
    parser.add_argument(
        "--invalid-state",
        type=str,
        default="X",
        help="Symbol to represent invalid or missing coordinate states.",
    )
    parser.add_argument(
        "--exclude-feat",
        type=int,
        default=None,
        help="One-based index of feature to exclude from prediction.",
    )
    args = parser.parse_args()

    encoder = torch.load(args.encoder, map_location="cpu")
    encoder.eval()
    centroids = np.loadtxt(args.centroids)
    virt_cb = (args.virt[0], args.virt[1], args.virt[2])

    for line in sys.stdin:
        fn = line.rstrip("\n")
        if not fn:
            continue
        try:
            basename, seq = process_pdb(
                fn,
                encoder,
                centroids,
                args.pdb_dir,
                virt_cb,
                args.invalid_state,
                args.exclude_feat,
            )
            print(f"{basename} {seq}")
        except Exception as e:
            print(f"Error processing {fn}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
