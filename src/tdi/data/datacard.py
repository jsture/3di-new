"""Data card generator: a human-readable DATACARD.md per processed dataset.

Derived entirely from manifest.json + report.json, so it never introduces facts
not already recorded by the pipeline.
"""

from typing import Any


def render_datacard(manifest: dict[str, Any], report: dict[str, Any]) -> str:
    """Render DATACARD.md from the manifest and report dicts.

    Args:
        manifest: Parsed manifest.json content.
        report: Parsed report.json content.

    Returns:
        Markdown string.
    """
    name = manifest.get("dataset_name", "unknown")
    pre = manifest.get("preprocessing", {})
    inputs = manifest.get("inputs", {})
    if "train" in report:
        stage = report["train"].get("stage_counts", {})
    else:
        stage = report.get("stage_counts", {})

    lines = [
        f"# Data Card — {name}",
        "",
        "## Source files",
        "",
    ]
    for label, rec in inputs.items():
        path = rec.get("path") if isinstance(rec, dict) else rec
        # sha256 may be absent (key missing) or explicitly None (file missing at build time);
        # `or ""` handles both so slicing never hits None.
        sha = (rec.get("sha256") or "")[:12] if isinstance(rec, dict) else ""
        lines.append(f"- **{label}**: `{path}` (sha256 `{sha}…`)")

    lines += [
        "",
        "## Filters & preprocessing",
        "",
        f"- Virtual center: {pre.get('virtual_center')}",
        f"- Sequence-delta convention: {pre.get('sequence_delta_convention')}",
        f"- Max Ca distance (A): {pre.get('max_ca_dist')}",
        f"- Max pairs per alignment: {pre.get('max_pairs_per_alignment')}",
        f"- Standardization: {pre.get('standardization')}",
        "",
        "## Counts",
        "",
        f"- Pairs before filters: {stage.get('n_pairs_before_filters')}",
        f"- After descriptor validity: {stage.get('n_pairs_after_descriptor_validity')}",
        f"- After Ca filter: {stage.get('n_pairs_after_ca_filter')}",
        f"- After max-pair cap: {stage.get('n_pairs_after_max_pairs')}",
        f"- Final bidirectional examples: {stage.get('n_final_examples')}",
        "",
        "## Feature definitions",
        "",
        "10-D residue descriptor: 7 inter-residue angle cosines, Ca-Ca distance, clipped "
        "sequence separation, and a signed log sequence-distance term "
        "(convention: partner index - source index).",
        "",
        "## Intended use",
        "",
        "Training/validation of the v2 VQ-VAE that learns a discrete 3Di-style alphabet.",
        "",
        "## Not intended for",
        "",
        "Benchmarking against other tools' CIGAR conventions (this dataset uses the custom "
        "`P`=aligned-pair convention) or as a held-out structural benchmark.",
        "",
    ]
    return "\n".join(lines)
