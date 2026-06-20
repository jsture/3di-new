#!/usr/bin/env python3
"""Standalone VQ-vs-FSQ comparison driver.

Runs the *normal* ``train`` + ``evaluate`` path twice -- once ``--quantizer vq``, once
``--quantizer fsq`` -- into ``<out-root>/ema_vq`` and ``<out-root>/fsq_5x4``, then tabulates the
two ``evaluation_report.json`` (+ ``train_log.csv``) side by side into ``comparison_report.json``
(and ``comparison.md``).

This is intentionally a separate, opt-in driver: the core ``tdi.v2`` train/evaluate/model code
has no knowledge of it (it must never import this script). You normally train one quantizer at a
time and only reach for this driver when you want the comparison. No sweeps -- exactly two runs.
"""

import argparse
import csv
import json
from pathlib import Path

from tdi.v2.cli import run_evaluate
from tdi.v2.train import train_model
from tdi.v2.train_config import load_train_config


def _final_val_loss(run_dir: Path) -> float | None:
    """Read the best (minimum) val_loss from ``train_log.csv`` if present.

    The training loop reloads the best (lowest-val_loss) checkpoint before export, so the
    figure that corresponds to the exported model is the minimum across epochs — not the
    last epoch, which under early stopping is worse than the model that was actually saved.
    """
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        return None
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return min((float(r["val_loss"]) for r in rows), default=None)


def _read_side(run_dir: Path) -> dict[str, object]:
    """Pull the comparison fields for a single run from its export + eval report."""
    with open(run_dir / "eval" / "evaluation_report.json") as f:
        report = json.load(f)
    return {
        "val_loss": _final_val_loss(run_dir),
        "state_entropy": report.get("normalized_entropy"),
        "dead_states": report.get("dead_state_fraction"),
        "aligned_mi": report.get("mi"),
        "mi_tot": report.get("mi_tot"),
        "submat_total_counts": report.get("total_counts"),
        "n_letters": report.get("n_letters"),
    }


def build_comparison(vq_dir: Path, fsq_dir: Path) -> dict[str, dict[str, object]]:
    """Assemble the side-by-side comparison table for two finished runs."""
    return {"ema_vq": _read_side(vq_dir), "fsq_5x4": _read_side(fsq_dir)}


def _write_markdown(comparison: dict[str, dict[str, object]], path: Path) -> None:
    """Render the comparison as a small Markdown table."""
    metrics = ["val_loss", "state_entropy", "dead_states", "aligned_mi", "mi_tot"]
    lines = ["# Quantizer comparison", "", "| metric | ema_vq | fsq_5x4 |", "| --- | --- | --- |"]
    for metric in metrics:
        vq = comparison["ema_vq"].get(metric)
        fsq = comparison["fsq_5x4"].get(metric)
        lines.append(f"| {metric} | {vq} | {fsq} |")
    path.write_text("\n".join(lines) + "\n")


def _train_and_evaluate(
    config_path: str,
    quantizer: str,
    out_dir: Path,
    pdb_dir: str,
    pairfile: str,
    virt: list[float] | None,
) -> None:
    """Run one normal train + evaluate into ``out_dir`` (and ``out_dir/eval``)."""
    cfg = load_train_config(
        config_path, {"model.quantizer": quantizer, "outputs.out_dir": str(out_dir)}
    )
    train_model(cfg)
    run_evaluate(
        argparse.Namespace(
            model_dir=str(out_dir),
            pdb_dir=pdb_dir,
            pairfile=pairfile,
            out_dir=str(out_dir / "eval"),
            virt=virt,
            invalid_state=None,
        )
    )


def main() -> None:
    """CLI entrypoint for the two-run comparison."""
    parser = argparse.ArgumentParser(description="Compare EMA-VQ vs FSQ on one train/eval setup.")
    parser.add_argument("--config", required=True, help="Train config (shared by both runs).")
    parser.add_argument("--pdb-dir", required=True, help="PDB directory for evaluation.")
    parser.add_argument("--pairfile", required=True, help="Alignment pairfile for evaluation.")
    parser.add_argument("--out-root", required=True, help="Root directory for both run dirs.")
    parser.add_argument(
        "--virt", type=float, nargs=3, default=None, help="Virtual center (alpha, beta, d)."
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    vq_dir = out_root / "ema_vq"
    fsq_dir = out_root / "fsq_5x4"

    for quantizer, out_dir in (("vq", vq_dir), ("fsq", fsq_dir)):
        _train_and_evaluate(
            args.config, quantizer, out_dir, args.pdb_dir, args.pairfile, args.virt
        )

    comparison = build_comparison(vq_dir, fsq_dir)
    with open(out_root / "comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2)
    _write_markdown(comparison, out_root / "comparison.md")
    print(f"Wrote comparison to {out_root / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
