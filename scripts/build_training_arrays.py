#!/usr/bin/env python3
"""Script to orchestrate feature extraction and processed dataset generation."""

import argparse
import sys

from tdi.data.pipeline import build_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract features from PDB files and build training feature arrays."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML data config (e.g. configs/data/scop_baseline.yaml).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow overwriting a populated output directory."
    )
    args = parser.parse_args()

    try:
        out_dir = build_features(args.config, overrides=None, force=args.force)
        print(f"Successfully wrote baseline training arrays to {out_dir}")
        return 0
    except Exception as e:
        print(f"Error building training arrays: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
