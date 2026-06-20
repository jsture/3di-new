"""Build-features orchestration for the tdi.data pipeline.

Produces an immutable, versioned processed-dataset directory: feature arrays,
scaler, per-pair metadata (with SCOP joins), structure QC, a preprocessing report,
a data card, and a manifest recording input/output hashes for reproducibility.
"""

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdi.data import datacard, report
from tdi.data.config import DataConfig, load_config
from tdi.data.hashing import array_record, git_commit, git_dirty, sha256_file
from tdi.data.scop import classify, load_scop_lookup
from tdi.data.structures import build_structures_table
from tdi.data.validate import validate_dataset
from tdi.v2.training_data import align_features, fit_standardizer
from tdi.v2.util import parse_pairfile_line

# Column order for per-pair metadata; also used to give an empty split a typed,
# column-bearing DataFrame (so downstream report/parquet code never sees a column-less frame).
# Lean-but-auditable: alignment_id encodes the source row (and split via the pairfile stem),
# so the standalone source_pairfile_row / SHA row_id are redundant and dropped. split_group_*
# records the leakage-grouping level actually used by scripts/make_splits.py (superfamily).
_METADATA_COLUMNS = [
    "alignment_id",
    "sid_source",
    "sid_target",
    "idx_source",
    "idx_target",
    "split_group_source",
    "split_group_target",
    "fold_source",
    "fold_target",
    "superfamily_source",
    "superfamily_target",
    "ca_dist_raw",
    "ca_dist_superposed",
]


def _read_pairfile(path: str | Path) -> list[tuple[int, str, str, str]]:
    """Read a pairfile into (source_row, sid1, sid2, cigar), sorted for determinism."""
    rows: list[tuple[int, str, str, str]] = []
    with open(path) as f:
        for source_row, line in enumerate(f):
            res = parse_pairfile_line(line)
            if res is not None:
                sid1, sid2, cigar = res
                rows.append((source_row, sid1, sid2, cigar))
    # Sort by content so iteration order is independent of file ordering.
    rows.sort(key=lambda r: (r[1], r[2], r[3], r[0]))
    return rows


def _split_metadata(
    meta: dict[str, Any],
    sid1: str,
    sid2: str,
    source_row: int,
    pairfile_stem: str,
    scop_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    """Expand one alignment's meta arrays into per-pair metadata records."""
    n = len(meta["idx_source"])
    # Enriched alignment_id encodes the pairfile (and thus the split), source row, and both
    # sids, so a metadata row maps back to its exact source line without a separate column.
    alignment_id = f"{pairfile_stem}:{source_row}:{sid1}:{sid2}"
    records: list[dict[str, Any]] = []
    for i in range(n):
        sid_src = meta["sid_source"][i]
        sid_tgt = meta["sid_target"][i]
        src_scop = classify(scop_lookup.get(sid_src))
        tgt_scop = classify(scop_lookup.get(sid_tgt))
        records.append(
            {
                "alignment_id": alignment_id,
                "sid_source": sid_src,
                "sid_target": sid_tgt,
                "idx_source": int(meta["idx_source"][i]),
                "idx_target": int(meta["idx_target"][i]),
                # Leakage-grouping level used by make_splits.py is superfamily.
                "split_group_source": src_scop["superfamily"],
                "split_group_target": tgt_scop["superfamily"],
                "fold_source": src_scop["fold"],
                "fold_target": tgt_scop["fold"],
                "superfamily_source": src_scop["superfamily"],
                "superfamily_target": tgt_scop["superfamily"],
                "ca_dist_raw": float(meta["ca_dist_raw"][i]),
                "ca_dist_superposed": float(meta["ca_dist_superposed"][i]),
            }
        )
    return records


def _process_split(
    cfg: DataConfig,
    pairfile: str,
    scop_lookup: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int], set[str], list[dict[str, Any]]]:
    """Process one split: returns (x, y, metadata, stage_counts, sids, skipped_records)."""
    alignments = _read_pairfile(pairfile)
    pairfile_stem = Path(pairfile).stem
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    sids: set[str] = set()
    counts = {
        "n_alignments_read": len(alignments),
        # Errored = true parse/extraction failures (also written to the skipped TSV).
        # Empty = alignments that legitimately filtered down to zero pairs.
        "n_alignments_errored": 0,
        "n_alignments_empty": 0,
        "n_pairs_before_filters": 0,
        "n_pairs_after_descriptor_validity": 0,
        "n_pairs_after_ca_filter": 0,
        "n_pairs_after_max_pairs": 0,
        "n_final_examples": 0,
        "n_alignments_dropped_degenerate_kabsch": 0,
        "n_pairs_dropped_degenerate_kabsch": 0,
        "degenerate_kabsch_alignment_ids_sample": [],
    }

    total = len(alignments)
    print(f"  Processing {total} alignments from {pairfile}...")
    for idx, (source_row, sid1, sid2, cigar) in enumerate(alignments):
        if (idx + 1) % 1000 == 0 or idx + 1 == total:
            print(
                f"    Progress: {idx + 1}/{total} alignments processed "
                f"({counts['n_alignments_errored']} errored, "
                f"{counts['n_alignments_empty']} empty)..."
            )
        # Record every referenced structure up front (even if its alignment errors or yields
        # no pairs) so the QC table can explain domains that produced no features.
        sids.update([sid1, sid2])
        try:
            x, y, meta = align_features(
                cfg.dataset.pdb_dir,
                cfg.features.virtual_center,
                sid1,
                sid2,
                cigar,
                max_ca_dist=cfg.features.max_ca_dist,
                max_pairs=cfg.sampling.max_pairs_per_alignment,
                seed=cfg.sampling.seed,
            )
        except Exception as exc:
            counts["n_alignments_errored"] += 1
            skipped_records.append(
                {
                    "source_row": source_row,
                    "sid1": sid1,
                    "sid2": sid2,
                    "cigar": cigar,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        counts["n_pairs_before_filters"] += meta.get("n_pairs_before_filters", 0)
        counts["n_pairs_after_descriptor_validity"] += meta.get(
            "n_pairs_after_descriptor_validity", 0
        )
        counts["n_pairs_after_ca_filter"] += meta.get("n_pairs_after_ca_filter", 0)
        counts["n_pairs_after_max_pairs"] += meta.get("n_pairs_after_max_pairs", 0)

        # Track degenerate Kabsch structural alignments
        if meta.get("kabsch_error") is not None:
            counts["n_alignments_dropped_degenerate_kabsch"] += 1
            counts["n_pairs_dropped_degenerate_kabsch"] += max(
                0,
                meta.get("n_pairs_after_descriptor_validity", 0)
                - meta.get("n_pairs_after_ca_filter", 0),
            )
            if len(counts["degenerate_kabsch_alignment_ids_sample"]) < 10:
                counts["degenerate_kabsch_alignment_ids_sample"].append(
                    f"{sid1}-{sid2} ({meta['kabsch_error']})"
                )

        if len(x) == 0:
            counts["n_alignments_empty"] += 1
            continue

        x_list.append(x)
        y_list.append(y)
        meta_records.extend(
            _split_metadata(meta, sid1, sid2, source_row, pairfile_stem, scop_lookup)
        )

    x_feat = np.vstack(x_list) if x_list else np.zeros((0, 10), dtype=np.float32)
    y_feat = np.vstack(y_list) if y_list else np.zeros((0, 10), dtype=np.float32)
    counts["n_final_examples"] = int(x_feat.shape[0])
    metadata = (
        pd.DataFrame.from_records(meta_records)
        if meta_records
        else pd.DataFrame(columns=_METADATA_COLUMNS)
    )

    # Fail policy is keyed on errored alignments (true failures), not on alignments that
    # legitimately filter down to zero pairs. The fraction threshold is the lenient guard;
    # fail_on_skipped_alignments is the strict "any failure aborts" switch.
    total_alignments = len(alignments)
    if total_alignments > 0:
        errored = counts["n_alignments_errored"]
        error_fraction = errored / total_alignments
        if cfg.preprocessing.fail_on_skipped_alignments:
            if errored > 0:
                raise RuntimeError(
                    f"fail_on_skipped_alignments is set and {errored} alignment(s) errored."
                )
        elif error_fraction > cfg.preprocessing.max_skipped_fraction:
            raise RuntimeError(
                f"Errored fraction {error_fraction:.3f} exceeds "
                f"max_skipped_fraction {cfg.preprocessing.max_skipped_fraction:.3f}."
            )

    return x_feat, y_feat, metadata, counts, sids, skipped_records


def build_features(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Run the full build-features pipeline and write the processed dataset.

    Args:
        config_path: Path to the YAML data config.
        overrides: Optional ``{"section.key": value}`` CLI overrides.
        force: If True, allow writing into a populated out_dir (default refuses).

    Returns:
        The output directory path.

    Raises:
        FileExistsError: If out_dir is non-empty and ``force`` is False.
    """
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path, overrides)
    out_dir = Path(cfg.outputs.out_dir)
    # Refuse any non-empty out_dir, not just one with a manifest: a crashed/partial build
    # leaves arrays without a manifest, and the next run would silently mix/overwrite them.
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{out_dir} is not empty; refusing to overwrite (pass force=True to override)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SCOP lookup table...")
    scop_lookup = load_scop_lookup(cfg.dataset.scop_lookup)

    # 1. Collect all referenced SIDs from the split pairfiles.
    print("Collecting referenced structure IDs from pairfiles...")
    train_alignments = _read_pairfile(cfg.dataset.train_pairfile)
    val_alignments = _read_pairfile(cfg.dataset.val_pairfile)
    referenced = set()
    for _row, sid1, sid2, _cigar in train_alignments + val_alignments:
        referenced.update([sid1, sid2])
    all_sids = sorted(referenced)

    # 2. Build the structures table exactly once.
    print(f"Building structure QC table for {len(all_sids)} unique domains (with cache checks)...")
    structures = build_structures_table(all_sids, cfg.dataset.pdb_dir)
    bad_structures = (structures["parse_status"] != "ok").sum()
    # Log progress for structure QC table building phase.
    print(
        f"Structure QC table built: {len(structures)} domains checked "
        f"({bad_structures} failed/missing)."
    )

    # 3. Early CIGAR validation if configured.
    if cfg.preprocessing.validate_cigars:
        print("Performing early CIGAR validation check...")
        validate_dataset(config_path, overrides=overrides, prebuilt_structures=structures)
        print("CIGAR validation check passed successfully.")

    print("Processing train split alignments...")
    x_train, y_train, train_meta, train_counts, _train_sids, train_skipped = _process_split(
        cfg, cfg.dataset.train_pairfile, scop_lookup
    )
    print("Processing validation split alignments...")
    x_val, y_val, val_meta, val_counts, _val_sids, val_skipped = _process_split(
        cfg, cfg.dataset.val_pairfile, scop_lookup
    )

    # Cast features to float32 to reduce memory footprint and avoid float64 overhead in training
    x_train = x_train.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)

    # Standardizer fit on train features only.
    print("Fitting feature standardizer scaler on train features...")
    mean, std = (
        fit_standardizer(x_train)
        if x_train.size
        else (
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        )
    )

    # Ensure mean and std are float32
    mean = mean.astype(np.float32, copy=False)
    std = std.astype(np.float32, copy=False)

    # Write arrays + scaler.
    print(f"Writing processed feature arrays and scaler to {out_dir}...")
    arrays = {
        "train_x_raw": x_train,
        "train_y_raw": y_train,
        "val_x_raw": x_val,
        "val_y_raw": y_val,
    }
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    np.savez(out_dir / "scaler.npz", mean=mean, std=std)

    # Pair metadata (one row per final example).
    print("Writing metadata parquets and skipped alignment lists...")
    train_meta.to_parquet(out_dir / "train_metadata.parquet", index=False)
    val_meta.to_parquet(out_dir / "val_metadata.parquet", index=False)

    # Write skipped alignments TSVs
    cols = ["source_row", "sid1", "sid2", "cigar", "error_type", "error"]
    df_train_skipped = pd.DataFrame(train_skipped, columns=cols)
    df_train_skipped.to_csv(out_dir / "train_skipped_alignments.tsv", sep="\t", index=False)

    df_val_skipped = pd.DataFrame(val_skipped, columns=cols)
    df_val_skipped.to_csv(out_dir / "val_skipped_alignments.tsv", sep="\t", index=False)

    # Structure-level QC across every referenced structure.
    structures.to_parquet(out_dir / "structures.parquet", index=False)

    # Report: one joint train/val report.json (the per-split duplicates are dropped). The
    # seq-separation and Ca-distance histograms are gated behind full_report (off by default).
    print("Compiling report, datacard, and writing manifest...")
    full_report = cfg.preprocessing.full_report
    report_dict = {
        "train": report.build_report(train_counts, x_train, train_meta, full_report=full_report),
        "val": report.build_report(val_counts, x_val, val_meta, full_report=full_report),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report_dict, f, indent=2, allow_nan=False)

    with open(out_dir / "report.md", "w") as f:
        f.write(report.render_report_md(report_dict, cfg.dataset.name))

    # Manifest with input + output hashes.
    manifest = _build_manifest(cfg, arrays, out_dir)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    # Data card derived from manifest + report.
    with open(out_dir / "DATACARD.md", "w") as f:
        f.write(datacard.render_datacard(manifest, report_dict))

    print("Preprocessing build completed successfully.")
    return out_dir


def _build_manifest(
    cfg: DataConfig, arrays: dict[str, np.ndarray], out_dir: Path
) -> dict[str, Any]:
    """Assemble the manifest dict (inputs, outputs, params, provenance)."""
    inputs: dict[str, Any] = {
        "pdb_dir": {"path": cfg.dataset.pdb_dir},  # per-structure hashes live in structures.parquet
    }
    for label, path in (
        ("train_pairfile", cfg.dataset.train_pairfile),
        ("val_pairfile", cfg.dataset.val_pairfile),
        ("scop_lookup", cfg.dataset.scop_lookup),
    ):
        if path and Path(path).exists():
            inputs[label] = {"path": path, "sha256": sha256_file(path)}
        elif path:
            inputs[label] = {"path": path, "sha256": None}

    outputs = {name: array_record(arr) for name, arr in arrays.items()}
    for name in (
        "train_metadata.parquet",
        "val_metadata.parquet",
        "structures.parquet",
        "scaler.npz",
        "train_skipped_alignments.tsv",
        "val_skipped_alignments.tsv",
    ):
        path = out_dir / name
        if path.exists():
            outputs[name] = {"sha256": sha256_file(path)}

    return {
        "dataset_name": cfg.dataset.name,
        "inputs": inputs,
        "outputs": outputs,
        "preprocessing": {
            "virtual_center": list(cfg.features.virtual_center),
            "sequence_delta_convention": cfg.features.sequence_delta_convention,
            "max_ca_dist": cfg.features.max_ca_dist,
            "max_pairs_per_alignment": cfg.sampling.max_pairs_per_alignment,
            "seed": cfg.sampling.seed,
            "standardization": "zscore_train_fit",
            "fail_on_skipped_alignments": cfg.preprocessing.fail_on_skipped_alignments,
            "max_skipped_fraction": cfg.preprocessing.max_skipped_fraction,
        },
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_hash": cfg.config_hash(),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
