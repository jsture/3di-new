"""Preprocessing report: a first-class artifact (report.json + report.md).

Aggregates the per-alignment stage counts and the pair metadata into stage-drop
reconciliation, feature statistics, and the cheap diagnostics the collaborator
wanted first: sequence-separation and Ca-distance histograms.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Sequence-separation bins: 1, 2-4, 5-12, 13-24, 25-64, >64.
_SEQ_SEP_EDGES = [1, 2, 5, 13, 25, 65]
_SEQ_SEP_LABELS = ["1", "2-4", "5-12", "13-24", "25-64", ">64"]

# Ca-distance bins (Angstroms) up to the typical max_ca_dist filter, plus overflow.
_CA_DIST_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


def _histogram_from_edges(values: np.ndarray, edges: list[float]) -> list[int]:
    """Count values into half-open bins [edges[i], edges[i+1])."""
    if values.size == 0:
        return [0] * (len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges)
    return [int(c) for c in counts]


def _ca_distance_bins(values: np.ndarray) -> list[dict[str, object]]:
    """Build strict-JSON Ca-distance bins, including explicit overflow."""
    bounded_counts = _histogram_from_edges(values, _CA_DIST_EDGES)
    bins = [
        {"label": f"[{_CA_DIST_EDGES[i]}, {_CA_DIST_EDGES[i + 1]})", "count": count}
        for i, count in enumerate(bounded_counts)
    ]
    overflow = int(np.sum(values >= _CA_DIST_EDGES[-1])) if values.size else 0
    bins.append({"label": f">={_CA_DIST_EDGES[-1]}", "count": overflow})
    return bins


def _seq_sep_histogram(features: np.ndarray) -> dict[str, int]:
    """Histogram of within-structure contact sequence separation |partner - source|.

    This is the genuine contact separation, recovered from the descriptor's signed
    log sequence-distance term (feature column 9 = sign(delta) * log(|delta| + 1)).
    It is NOT |idx_target - idx_source| from the metadata: those indices live in two
    different structures, so their difference is a cross-structure alignment offset,
    not a sequence separation.
    """
    if features.size == 0 or features.shape[1] < 10:
        return dict.fromkeys(_SEQ_SEP_LABELS, 0)
    # Invert sign(delta) * log(|delta| + 1) to recover |delta| (the contact separation).
    sep = np.rint(np.exp(np.abs(features[:, 9])) - 1.0).astype(int)
    # np.digitize with the lower edges maps each value to its bin index.
    bin_idx = np.digitize(sep, _SEQ_SEP_EDGES, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(_SEQ_SEP_LABELS) - 1)
    counts = np.bincount(bin_idx, minlength=len(_SEQ_SEP_LABELS))
    return {label: int(counts[i]) for i, label in enumerate(_SEQ_SEP_LABELS)}


def build_report(
    stage_counts: dict[str, int],
    features: np.ndarray,
    metadata: pd.DataFrame,
    full_report: bool = False,
) -> dict[str, object]:
    """Assemble the report dict.

    Args:
        stage_counts: Aggregated per-stage pair counts (rows read/skipped, before/after
            each filter) plus ``n_final_examples``.
        features: Final stacked input feature array of shape (N, D).
        metadata: Pair metadata table (one row per final example).
        full_report: If True, also include the seq-separation and Ca-distance histograms
            (off by default to keep the report lean).

    Returns:
        JSON-serializable report dict.
    """
    ca_dist = (
        metadata["ca_dist_superposed"].dropna().to_numpy()
        if "ca_dist_superposed" in metadata
        else np.empty(0)
    )

    def _level_counts(column: str) -> dict[str, int]:
        if column not in metadata or metadata.empty:
            return {}
        return {str(k): int(v) for k, v in metadata[column].value_counts().items()}

    def _alignment_quantiles() -> dict[str, float]:
        if "alignment_id" not in metadata or metadata.empty:
            return {}
        counts = metadata["alignment_id"].value_counts().to_numpy()
        if len(counts) == 0:
            return {}
        return {
            "min": float(counts.min()),
            "p25": float(np.percentile(counts, 25)),
            "median": float(np.percentile(counts, 50)),
            "p75": float(np.percentile(counts, 75)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "max": float(counts.max()),
            "mean": float(counts.mean()),
            "std": float(counts.std()),
            "count": len(counts),
        }

    feat_stats = {
        "mean": features.mean(axis=0).tolist() if features.size else [],
        "std": features.std(axis=0).tolist() if features.size else [],
        "min": features.min(axis=0).tolist() if features.size else [],
        "max": features.max(axis=0).tolist() if features.size else [],
    }

    report: dict[str, object] = {
        "stage_counts": stage_counts,
        "feature_stats": feat_stats,
        "examples_per_fold": _level_counts("fold_source"),
        "examples_per_superfamily": _level_counts("superfamily_source"),
        "examples_per_alignment": _alignment_quantiles(),
    }
    if full_report:
        report["sequence_separation_histogram"] = _seq_sep_histogram(features)
        report["ca_distance_histogram"] = {"bins": _ca_distance_bins(ca_dist)}
    return report


def reconcile(stage_counts: dict[str, int]) -> bool:
    """Check the stage counts are internally consistent.

    Two independent invariants (the previous telescoping subtraction was a tautology
    that always returned True):
      1. Each filter can only remove pairs, so the per-stage counts must be
         non-increasing: before >= after_validity >= after_ca >= after_cap >= 0.
      2. Bidirectional mirroring doubles the post-cap pair count into the final example
         count: n_pairs_after_max_pairs * 2 == n_final_examples.
    """
    read = stage_counts.get("n_pairs_before_filters", 0)
    after_validity = stage_counts.get("n_pairs_after_descriptor_validity", 0)
    after_ca = stage_counts.get("n_pairs_after_ca_filter", 0)
    after_cap = stage_counts.get("n_pairs_after_max_pairs", 0)
    final = stage_counts.get("n_final_examples", 0)

    monotonic = read >= after_validity >= after_ca >= after_cap >= 0
    mirror_ok = after_cap * 2 == final
    return monotonic and mirror_ok


def render_report_md(report_dict: Mapping[str, object], dataset_name: str) -> str:
    """Render the report dict as Markdown."""
    if "train" in report_dict and "val" in report_dict:
        # Render a joint train and validation split report.
        lines = [
            f"# Preprocessing Report — {dataset_name}",
            "",
        ]
        for split in ("train", "val"):
            split_report = report_dict[split]
            assert isinstance(split_report, dict)
            sc = split_report["stage_counts"]
            assert isinstance(sc, dict)
            lines += [
                f"## {split.capitalize()} Split",
                "",
                "### Stage counts",
                "",
                "| Stage | Pairs |",
                "| --- | ---: |",
            ]
            for key, val in sc.items():
                lines.append(f"| {key} | {val} |")

            # Histograms are present only when the build ran with --full-report.
            if "sequence_separation_histogram" in split_report:
                seq_hist = split_report["sequence_separation_histogram"]
                assert isinstance(seq_hist, dict)
                lines += [
                    "",
                    "### Sequence-separation histogram",
                    "",
                    "| Bin | Count |",
                    "| --- | ---: |",
                ]
                lines += [f"| {k} | {v} |" for k, v in seq_hist.items()]

            if "ca_distance_histogram" in split_report:
                ca_hist = split_report["ca_distance_histogram"]
                assert isinstance(ca_hist, dict)
                lines += [
                    "",
                    "### Ca-distance histogram (A)",
                    "",
                    "| Bin | Count |",
                    "| --- | ---: |",
                ]
                for bin_record in ca_hist["bins"]:
                    lines.append(f"| {bin_record['label']} | {bin_record['count']} |")

            lines += [
                "",
                f"Stage reconciliation: {'OK' if reconcile(sc) else 'MISMATCH'}",
                "",
                "---",
                "",
            ]
        return "\n".join(lines)
    else:
        # Fallback to single split rendering
        sc = report_dict["stage_counts"]
        assert isinstance(sc, dict)
        lines = [
            f"# Preprocessing Report — {dataset_name}",
            "",
            "## Stage counts",
            "",
            "| Stage | Pairs |",
            "| --- | ---: |",
        ]
        for key, val in sc.items():
            lines.append(f"| {key} | {val} |")

        if "sequence_separation_histogram" in report_dict:
            seq_hist = report_dict["sequence_separation_histogram"]
            assert isinstance(seq_hist, dict)
            lines += [
                "",
                "## Sequence-separation histogram",
                "",
                "| Bin | Count |",
                "| --- | ---: |",
            ]
            lines += [f"| {k} | {v} |" for k, v in seq_hist.items()]

        if "ca_distance_histogram" in report_dict:
            ca_hist = report_dict["ca_distance_histogram"]
            assert isinstance(ca_hist, dict)
            lines += ["", "## Ca-distance histogram (A)", "", "| Bin | Count |", "| --- | ---: |"]
            for bin_record in ca_hist["bins"]:
                lines.append(f"| {bin_record['label']} | {bin_record['count']} |")

        lines += ["", f"Stage reconciliation: {'OK' if reconcile(sc) else 'MISMATCH'}", ""]
        return "\n".join(lines)
