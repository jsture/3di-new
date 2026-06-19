#!/usr/bin/env python3
"""CLI: extract features from aligned structure pairs to compile VQ-VAE training data."""

import argparse
import json
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
    parser.add_argument(
        "--seed", type=int, default=123, help="Seed value for sub-sampling reproducibility."
    )
    args = parser.parse_args()

    # Load allowed SIDs from manifest
    allowed_sids = set()
    sid2group = {}
    if args.manifest:
        df_manifest = pd.read_csv(args.manifest)
        allowed_sids = set(df_manifest["structure_id"].values)
        sid2group = dict(zip(df_manifest["structure_id"], df_manifest["group_id"]))
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
    cap_records = []

    # Report accumulators
    n_pairs_before_filters_total = 0
    n_pairs_after_descriptor_validity_total = 0
    n_pairs_after_ca_filter_total = 0
    n_pairs_after_max_pairs_total = 0

    for sid1, sid2, cigar_string in alignments:
        try:
            x, y, meta = align_features(
                args.pdb_dir,
                virtual_center,
                sid1,
                sid2,
                cigar_string,
                max_ca_dist=args.max_ca_dist,
                max_pairs=args.max_pairs,
                seed=args.seed,
            )

            # Track filter counts
            n_pairs_before_filters_total += meta.get("n_pairs_before_filters", 0)
            n_pairs_after_descriptor_validity_total += meta.get(
                "n_pairs_after_descriptor_validity", 0
            )
            n_pairs_after_ca_filter_total += meta.get("n_pairs_after_ca_filter", 0)
            n_pairs_after_max_pairs_total += meta.get("n_pairs_after_max_pairs", 0)

            # Track pair cap info
            cap_records.append(
                {
                    "alignment_id": f"{sid1}-{sid2}",
                    "n_pairs_before_cap": meta.get("n_pairs_after_ca_filter", 0),
                    "n_pairs_after_cap": meta.get("n_pairs_after_max_pairs", 0),
                    "cap_seed": meta.get("cap_seed"),
                }
            )

            if len(x) > 0:
                xy_list.append((x, y))

                # Expand meta dict to rows
                for i in range(len(x)):
                    row = {
                        "sid_source": meta["sid_source"][i],
                        "sid_target": meta["sid_target"][i],
                        "idx_source": meta["idx_source"][i],
                        "idx_target": meta["idx_target"][i],
                        "ca_dist_superposed": (
                            meta["ca_dist_superposed"][i]
                            if meta["ca_dist_superposed"] is not None
                            else None
                        ),
                        "ca_dist_raw": meta["ca_dist_raw"][i],
                        "group_source": sid2group.get(meta["sid_source"][i], meta["sid_source"][i]),
                    }
                    meta_list.append(row)
        except Exception as e:
            print(f"Error processing {sid1}-{sid2}: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save cap records
    df_caps = pd.DataFrame(cap_records)
    df_caps.to_csv(out_dir / "alignment_pair_caps.csv", index=False)

    if xy_list:
        x_feat = np.vstack([x for x, y in xy_list])
        y_feat = np.vstack([y for x, y in xy_list])

        # We save x and y explicitly instead of dstack for clarity, but maintaining legacy behavior.
        # Since v2 train script uses training_data_raw[:, :, 0], we stick to dstack.
        stacked = np.dstack([x_feat, y_feat])
        np.save(out_dir / "data.npy", stacked)

        # Save metadata
        df_meta = pd.DataFrame(meta_list)
        df_meta.to_parquet(out_dir / "metadata.parquet", index=False)

        # Compute report statistics
        unique_sids = set(df_meta["sid_source"].unique()) | set(df_meta["sid_target"].unique())
        examples_per_alignment = [len(x) for x, y in xy_list]

        # Histogram for examples per alignment
        align_hist_counts, align_hist_bins = np.histogram(
            examples_per_alignment, bins=[0, 10, 50, 100, 200, 500, 1000]
        )

        # Histogram for examples per source group
        group_counts = df_meta["group_source"].value_counts().tolist()
        group_hist_counts, group_hist_bins = np.histogram(
            group_counts, bins=[0, 10, 50, 100, 500, 1000, 5000, 10000]
        )

        # Quantiles of superposed CA distance
        non_null_dists = df_meta["ca_dist_superposed"].dropna().values
        if len(non_null_dists) > 0:
            q_vals = [0, 25, 50, 75, 90, 95, 99, 100]
            q_results = np.percentile(non_null_dists, q_vals)
            ca_dist_quantiles = {f"{q}%": float(v) for q, v in zip(q_vals, q_results)}
        else:
            ca_dist_quantiles = {}

        # Feature stats
        feat_mean = x_feat.mean(axis=0).tolist()
        feat_std = x_feat.std(axis=0).tolist()
        feat_min = x_feat.min(axis=0).tolist()
        feat_max = x_feat.max(axis=0).tolist()
        nan_count = int(np.isnan(x_feat).sum())
        inf_count = int(np.isinf(x_feat).sum())

        report = {
            "n_structures": len(unique_sids),
            "n_alignments": len(xy_list),
            "n_pairs_before_filters": n_pairs_before_filters_total,
            "n_pairs_after_descriptor_validity": n_pairs_after_descriptor_validity_total,
            "n_pairs_after_ca_filter": n_pairs_after_ca_filter_total,
            "n_pairs_after_max_pairs": n_pairs_after_max_pairs_total,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "feature_min": feat_min,
            "feature_max": feat_max,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "examples_per_alignment_histogram": {
                "bins": align_hist_bins.tolist(),
                "counts": align_hist_counts.tolist(),
            },
            "examples_per_source_group_histogram": {
                "bins": group_hist_bins.tolist(),
                "counts": group_hist_counts.tolist(),
            },
            "ca_dist_superposed_quantiles": ca_dist_quantiles,
        }

        with open(out_dir / "training_data_report.json", "w") as f_rep:
            json.dump(report, f_rep, indent=2)

        print(f"Generated {len(x_feat)} pairs from {len(xy_list)} alignments.")
        print(f"Saved artifacts to {out_dir}")
    else:
        print("Warning: No alignments produced valid pairs. Output not written.")


if __name__ == "__main__":
    main()
