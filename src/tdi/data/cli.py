"""Command-line interface for the tdi.data preprocessing pipeline.

Usage: ``python -m tdi.data <build-features|validate|report> --config ...``.
CLI flags override the YAML config; the resolved config is recorded in the manifest.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from tdi.data.config import load_config
from tdi.data.pipeline import build_features
from tdi.data.report import render_report_md
from tdi.data.validate import validate_dataset


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-None CLI overrides as a ``{"section.key": value}`` map."""
    return {
        "outputs.out_dir": args.out_dir,
        "features.max_ca_dist": args.max_ca_dist,
        "sampling.max_pairs_per_alignment": args.max_pairs,
        "sampling.seed": args.seed,
        "features.virtual_center": (
            tuple(args.virtual_center) if args.virtual_center is not None else None
        ),
        "preprocessing.fail_on_skipped_alignments": args.fail_on_skipped,
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by all subcommands."""
    parser.add_argument("--config", required=True, help="Path to the YAML data config.")
    parser.add_argument("--out_dir", default=None, help="Override outputs.out_dir.")
    parser.add_argument("--max_ca_dist", type=float, default=None, help="Override max Ca dist.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Override max pairs/alignment.")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling seed.")
    parser.add_argument(
        "--virtual_center",
        type=float,
        nargs=3,
        default=None,
        help="Override features.virtual_center (three floats).",
    )
    parser.add_argument(
        "--fail_on_skipped",
        action="store_true",
        default=None,
        help="Override preprocessing.fail_on_skipped_alignments to True.",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m tdi.data``."""
    parser = argparse.ArgumentParser(
        prog="tdi.data", description="tdi.data preprocessing pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-features", help="Extract features, write arrays + metadata.")
    _add_common(p_build)
    p_build.add_argument("--force", action="store_true", help="Allow overwriting a populated dir.")

    p_validate = sub.add_parser("validate", help="Structure QC + CIGAR-semantics checks.")
    _add_common(p_validate)

    p_report = sub.add_parser("report", help="(Re)render report.md from an out_dir report.json.")
    _add_common(p_report)

    args = parser.parse_args(argv)

    if args.command == "build-features":
        out_dir = build_features(args.config, _overrides(args), force=args.force)
        print(f"Wrote processed dataset to {out_dir}")
        return 0

    if args.command == "validate":
        summary = validate_dataset(args.config, _overrides(args))
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "report":
        cfg = load_config(args.config, _overrides(args))
        out_dir = Path(cfg.outputs.out_dir)
        with open(out_dir / "report.json") as f:
            report_dict = json.load(f)
        md = render_report_md(report_dict, cfg.dataset.name)
        (out_dir / "report.md").write_text(md)
        print(f"Rendered {out_dir / 'report.md'}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
