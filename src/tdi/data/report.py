"""Preprocessing report: a first-class artifact (report.json + report.md).

Aggregates the per-alignment stage counts and the pair metadata into stage-drop
reconciliation, feature statistics, and the cheap diagnostics the collaborator
wanted first: sequence-separation and Ca-distance histograms.
"""

import numpy as np
import pandas as pd

# Sequence-separation bins: 1, 2-4, 5-12, 13-24, 25-64, >64.
_SEQ_SEP_EDGES = [1, 2, 5, 13, 25, 65]
_SEQ_SEP_LABELS = ["1", "2-4", "5-12", "13-24", "25-64", ">64"]

# Ca-distance bins (Angstroms) up to the typical max_ca_dist filter, plus overflow.
_CA_DIST_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, float("inf")]


def _histogram_from_edges(values: np.ndarray, edges: list[float]) -> list[int]:
    """Count values into half-open bins [edges[i], edges[i+1])."""
    if values.size == 0:
        return [0] * (len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges)
    return [int(c) for c in counts]


def _seq_sep_histogram(metadata: pd.DataFrame) -> dict[str, int]:
    """Histogram of |idx_target - idx_source| over the labelled separation bins."""
    if metadata.empty:
        return dict.fromkeys(_SEQ_SEP_LABELS, 0)
    sep = (metadata["idx_target"] - metadata["idx_source"]).abs().to_numpy()
    # np.digitize with the lower edges maps each value to its bin index.
    bin_idx = np.digitize(sep, _SEQ_SEP_EDGES, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(_SEQ_SEP_LABELS) - 1)
    counts = np.bincount(bin_idx, minlength=len(_SEQ_SEP_LABELS))
    return {label: int(counts[i]) for i, label in enumerate(_SEQ_SEP_LABELS)}


def build_report(
    stage_counts: dict[str, int],
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Assemble the report dict.

    Args:
        stage_counts: Aggregated per-stage pair counts (rows read/skipped, before/after
            each filter) plus ``n_final_examples``.
        features: Final stacked input feature array of shape (N, D).
        metadata: Pair metadata table (one row per final example).

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

    feat_stats = {
        "mean": features.mean(axis=0).tolist() if features.size else [],
        "std": features.std(axis=0).tolist() if features.size else [],
        "min": features.min(axis=0).tolist() if features.size else [],
        "max": features.max(axis=0).tolist() if features.size else [],
    }

    return {
        "stage_counts": stage_counts,
        "feature_stats": feat_stats,
        "sequence_separation_histogram": _seq_sep_histogram(metadata),
        "ca_distance_histogram": {
            "edges": _CA_DIST_EDGES,
            "counts": _histogram_from_edges(ca_dist, _CA_DIST_EDGES),
        },
        "examples_per_fold": _level_counts("fold_source"),
        "examples_per_superfamily": _level_counts("superfamily_source"),
        "examples_per_alignment": _level_counts("alignment_id"),
    }


def reconcile(stage_counts: dict[str, int]) -> bool:
    """Check stage drops reconcile: rows_read - sum(drops) == n_final pairs (pre-mirror).

    The final bidirectional example count is twice the post-cap pair count, so we
    reconcile against the pre-mirror pair count.
    """
    read = stage_counts.get("n_pairs_before_filters", 0)
    after_cap = stage_counts.get("n_pairs_after_max_pairs", 0)
    drop_validity = read - stage_counts.get("n_pairs_after_descriptor_validity", 0)
    drop_ca = stage_counts.get("n_pairs_after_descriptor_validity", 0) - stage_counts.get(
        "n_pairs_after_ca_filter", 0
    )
    drop_cap = stage_counts.get("n_pairs_after_ca_filter", 0) - after_cap
    return read - (drop_validity + drop_ca + drop_cap) == after_cap


def render_report_md(report: dict[str, object], dataset_name: str) -> str:
    """Render the report dict as Markdown."""
    sc = report["stage_counts"]
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

    seq_hist = report["sequence_separation_histogram"]
    assert isinstance(seq_hist, dict)
    lines += ["", "## Sequence-separation histogram", "", "| Bin | Count |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in seq_hist.items()]

    ca_hist = report["ca_distance_histogram"]
    assert isinstance(ca_hist, dict)
    lines += ["", "## Ca-distance histogram (A)", "", "| Bin | Count |", "| --- | ---: |"]
    edges, counts = ca_hist["edges"], ca_hist["counts"]
    for i, c in enumerate(counts):
        lines.append(f"| [{edges[i]}, {edges[i + 1]}) | {c} |")

    lines += ["", f"Stage reconciliation: {'OK' if reconcile(sc) else 'MISMATCH'}", ""]
    return "\n".join(lines)
