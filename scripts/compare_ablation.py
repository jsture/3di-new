#!/usr/bin/env python3
"""Tabulate the continuous-bypass ablation against its quantized controls.

Reads ``train_log.csv`` from each arm of the ablation and reports the best validation
reconstruction loss. The bypass arm is the reference: the percentage column is how much
each quantized arm pays, relative to it, for having a discrete bottleneck at all.
"""

import argparse
import csv
from pathlib import Path

# (label, default run directory). The bypass must stay first: it is the comparison baseline.
ARMS = [
    ("continuous (bypass)", "runs/ablation/continuous"),
    ("vq + k-means", "runs/ablation/vq_kmeans"),
    ("vq random init", "runs/ablation/vq_nokmeans"),
]


def _summarize(run_dir: Path) -> dict[str, float] | None:
    """Read one arm's log into best/first val_loss and the epoch that achieved it."""
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    val = [float(row["val_loss"]) for row in rows]
    best_index = val.index(min(val))
    return {
        "epochs": len(rows),
        "best_epoch": best_index,
        "best_val": min(val),
        # How much the run improved on its own first epoch -- near zero means the run was
        # decided at initialization and further training bought nothing.
        "gain": val[0] - min(val),
        "perplexity": float(rows[best_index].get("perplexity", "nan")),
    }


def main() -> None:
    """Print the ablation comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root holding the run dirs.")
    args = parser.parse_args()
    root = Path(args.root)

    summaries = [(label, _summarize(root / run_dir)) for label, run_dir in ARMS]
    baseline = next((s["best_val"] for label, s in summaries if s and "bypass" in label), None)

    header = f"{'arm':22s} {'best_val':>10s} {'ep':>3s} {'run gain':>10s} {'perplex':>9s}"
    if baseline is not None:
        header += f" {'vs bypass':>10s}"
    print(header)
    print("-" * len(header))

    for label, summary in summaries:
        if summary is None:
            print(f"{label:22s} {'(no run)':>10s}")
            continue
        line = (
            f"{label:22s} {summary['best_val']:10.5f} {summary['best_epoch']:3d} "
            f"{summary['gain']:+10.5f} {summary['perplexity']:9.2f}"
        )
        if baseline is not None:
            line += f" {100 * (summary['best_val'] - baseline) / baseline:+9.2f}%"
        print(line)


if __name__ == "__main__":
    main()
