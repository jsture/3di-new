#!/usr/bin/env python3
"""Evaluate every run under a directory and compile the alphabet metrics side by side.

Training is scored on validation reconstruction loss, but the deliverable is scored on
mutual information between aligned states. Those are different quantities and their
relationship is unmeasured, so this script puts them in the same table: each run's best
``val_loss`` next to the ``mi``/``mi_tot`` its exported alphabet actually achieves.

Runs that form no alphabet (the continuous-bypass ablation) are reported as skipped rather
than silently omitted -- their absence from the ranking is itself information.

Usage::

    uv run python scripts/evaluate_runs.py --runs-root runs/ablation
"""

import argparse
import contextlib
import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import tdi.v2.model
from tdi.v2.cli import run_evaluate
from tdi.v2.train import _read_provenance

# The transition-adjustment constant baked into calc_alphabet_mi; used here only to back the
# chain-autocorrelation term back out of the reported pair, never to recompute mi_tot.
_MI_PREV_WEIGHT = 1.0 - 0.057


@dataclass
class RunResult:
    """One run's training and alphabet metrics, joined on the run directory."""

    name: str
    quantizer: str
    n_states: int
    z_dim: int
    # Training side: what early stopping actually selected on.
    best_val_loss: float | None
    best_epoch: int | None
    epochs: int | None
    # Alphabet side: what the exported model scores on held-out alignments.
    mi: float | None = None
    mi_tot: float | None = None
    mi_prev: float | None = None
    chain_fraction: float | None = None
    normalized_entropy: float | None = None
    dead_state_fraction: float | None = None
    n_sequences: int | None = None
    failure_rate: float | None = None
    top_state_share: float | None = None
    skipped: str | None = None


# ---------------------------------------------------------------------------
# Architecture compatibility
#
# Exports record the quantizer and the dims but not which MLP built them, so a checkpoint
# written before an architecture swap cannot be reloaded afterwards. Rather than requiring
# the tree to be reverted to read old runs, both shapes are defined here and the right one
# is selected from the checkpoint's own parameter names.
# ---------------------------------------------------------------------------


class _ResidualMLP(nn.Module):
    """Residual blocks with LayerNorm + SiLU. Keys: ``input.*``, ``blocks.N.*``, ``output.*``."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


class _SimpleMLP(nn.Module):
    """Plain stacked Linear + SiLU. Keys: ``net.N.*`` only."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _detect_architecture(run_dir: Path) -> tuple[str, type[nn.Module]]:
    """Pick the MLP class matching a run's exported encoder, by inspecting its parameter names."""
    encoder_path = run_dir / "encoder_state_dict.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"No encoder_state_dict.pt in {run_dir}")
    keys = list(torch.load(encoder_path, map_location="cpu", weights_only=True).keys())
    if any(key.startswith("blocks.") for key in keys):
        return "residual", _ResidualMLP
    if any(key.startswith("net.") for key in keys):
        return "simple", _SimpleMLP
    raise ValueError(f"Unrecognized encoder layout in {run_dir}: {keys[:4]}")


def _mlp_class_names() -> list[str]:
    """Names of MLP classes defined in ``tdi.v2.model``, whatever they are currently called.

    AlphabetModel builds its encoder and decoder from a module-level class, but that class
    gets renamed as architectures are swapped, so hardcoding one name silently no-ops. The
    quantizers are imported into the same namespace, so restrict to classes actually defined
    in that module and exclude AlphabetModel itself.
    """
    names = []
    for name in dir(tdi.v2.model):
        attribute = getattr(tdi.v2.model, name)
        if (
            isinstance(attribute, type)
            and issubclass(attribute, nn.Module)
            and attribute is not tdi.v2.model.AlphabetModel
            and attribute.__module__ == tdi.v2.model.__name__
        ):
            names.append(name)
    return names


@contextlib.contextmanager
def _architecture(mlp_class: type[nn.Module]) -> Iterator[None]:
    """Temporarily point every MLP class in ``tdi.v2.model`` at ``mlp_class``.

    A run used one architecture for both encoder and decoder, so rebinding all of them is
    safe and removes any dependence on what the current class happens to be named.
    """
    names = _mlp_class_names()
    if not names:
        raise RuntimeError("No MLP class found in tdi.v2.model to substitute.")
    originals = {name: getattr(tdi.v2.model, name) for name in names}
    try:
        for name in names:
            setattr(tdi.v2.model, name, mlp_class)
        yield
    finally:
        for name, original in originals.items():
            setattr(tdi.v2.model, name, original)


def _discover_runs(runs_root: Path) -> list[Path]:
    """Return every immediate subdirectory that looks like a model export."""
    if not runs_root.is_dir():
        raise NotADirectoryError(f"Runs root not found: {runs_root}")
    return sorted(
        path for path in runs_root.iterdir() if path.is_dir() and (path / "config.json").exists()
    )


def _read_training_metrics(run_dir: Path) -> tuple[float | None, int | None, int | None]:
    """Read (best val_loss, epoch achieving it, epochs run) from ``train_log.csv``."""
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        return None, None, None
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, None
    losses = [float(row["val_loss"]) for row in rows]
    return min(losses), losses.index(min(losses)), len(rows)


def _resolve_virtual_center(run_dir: Path, override: list[float] | None) -> list[float] | None:
    """Find the virtual center for a run: CLI override, then export, then dataset manifest.

    Older exports recorded ``virtual_center: null`` because the training run could not read
    it from the processed dataset. Rather than fail those runs, walk the same provenance
    chain the trainer uses and recover it from the dataset that produced them.
    """
    if override is not None:
        return override

    with open(run_dir / "config.json") as f:
        exported = json.load(f)
    if exported.get("virtual_center") is not None:
        center: list[float] = exported["virtual_center"]
        return center

    resolved_path = run_dir / "run_config.resolved.json"
    if not resolved_path.exists():
        return None
    with open(resolved_path) as f:
        resolved = json.load(f)
    processed_dir = resolved.get("data", {}).get("processed_dir")
    if processed_dir is None:
        return None
    virtual_center, _ = _read_provenance(Path(processed_dir))
    return virtual_center


def _evaluate_one(
    run_dir: Path, pdb_dir: str, pairfile: str, virt: list[float] | None, force: bool
) -> Path | None:
    """Run evaluation into ``<run_dir>/eval`` unless a report already exists.

    Returns the report path, or None when the run forms no alphabet.
    """
    eval_dir = run_dir / "eval"
    report_path = eval_dir / "evaluation_report.json"
    if report_path.exists() and not force:
        print(f"  reusing existing report ({report_path})")
        return report_path

    _, mlp_class = _detect_architecture(run_dir)
    with _architecture(mlp_class):
        run_evaluate(
            argparse.Namespace(
                model_dir=str(run_dir),
                pdb_dir=pdb_dir,
                pairfile=pairfile,
                out_dir=str(eval_dir),
                virt=_resolve_virtual_center(run_dir, virt),
                invalid_state=None,
                # One unreadable structure should not abort a batch of runs; the per-run
                # failure_rate is reported in the table so a bad arm stays visible.
                max_failure_rate=1.0,
            )
        )
    return report_path


def _collect(run_dir: Path, pdb_dir: str, pairfile: str, virt: list[float] | None, force: bool):
    """Evaluate one run and join its alphabet metrics onto its training metrics."""
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    best_val, best_epoch, epochs = _read_training_metrics(run_dir)

    result = RunResult(
        name=run_dir.name,
        quantizer=config.get("quantizer", "?"),
        n_states=int(config.get("n_states", 0)),
        z_dim=int(config.get("z_dim", 0)),
        best_val_loss=best_val,
        best_epoch=best_epoch,
        epochs=epochs,
    )

    print(f"[{run_dir.name}] quantizer={result.quantizer} n_states={result.n_states}")
    try:
        report_path = _evaluate_one(run_dir, pdb_dir, pairfile, virt, force)
    except Exception as exc:
        # The bypass ablation raises by design; anything else is a real failure worth seeing.
        result.skipped = " ".join(str(exc).split())[:160]
        print(f"  skipped: {result.skipped}")
        return result

    if report_path is None or not report_path.exists():
        result.skipped = "no evaluation report produced"
        return result

    return _read_eval_report(report_path, result)


def _read_eval_report(report_path: Path, result: RunResult) -> RunResult:
    """Copy an evaluation report's alphabet metrics onto a run row.

    Split out from ``_collect`` so the mapping from report keys to table columns can be
    tested without running an evaluation. ``mi_prev`` and ``chain_fraction`` are read rather
    than recomputed whenever the evaluator emitted them; the table falls back to deriving
    them only for runs scored before they existed.
    """
    with open(report_path) as f:
        report = json.load(f)
    usage = report.get("state_usage", [])
    total_usage = sum(usage)

    result.mi = report.get("mi")
    result.mi_tot = report.get("mi_tot")
    result.mi_prev = report.get("mi_prev")
    result.chain_fraction = report.get("chain_fraction")
    result.normalized_entropy = report.get("normalized_entropy")
    result.dead_state_fraction = report.get("dead_state_fraction")
    result.n_sequences = report.get("n_sequences")
    result.failure_rate = report.get("failure_rate")
    result.top_state_share = max(usage) / total_usage if total_usage > 0 else None
    return result


def _effective_states(result: RunResult) -> float | None:
    """Convert normalized entropy back to an effective state count, 2**H_bits.

    ``normalized_entropy`` is reported in nats over ln(n_states), so scaling it by
    log2(n_states) recovers the entropy in bits.
    """
    if result.normalized_entropy is None or result.n_states < 2:
        return None
    entropy_bits = result.normalized_entropy * math.log2(result.n_states)
    return 2.0**entropy_bits


def _chain_autocorrelation(result: RunResult) -> float | None:
    """The lagged-MI term that ``mi_tot`` subtracts from ``mi``.

    This is how much of the raw MI is explained by a state predicting its own sequence
    neighbour -- local smoothness the alphabet gets for free rather than alignment signal.
    Taken from the report when present, and otherwise backed out of the reported pair so
    runs evaluated before ``mi_prev`` was emitted still tabulate.
    """
    if result.mi_prev is not None:
        return result.mi_prev
    if result.mi is None or result.mi_tot is None:
        return None
    return (result.mi - result.mi_tot) / _MI_PREV_WEIGHT


def _chain_fraction(result: RunResult) -> float | None:
    """Share of raw MI removed by the transition adjustment.

    The diagnostic that separates real alignment signal from an alphabet inflating raw MI
    with longer correlated state runs: a soft-MI arm whose ``mi`` rises while this rises
    with it has bought nothing.
    """
    if result.chain_fraction is not None:
        return result.chain_fraction
    if result.mi is None or result.mi_tot is None or result.mi == 0.0:
        return None
    return (result.mi - result.mi_tot) / result.mi


def _print_table(results: list[RunResult]) -> None:
    """Print the joined training/alphabet table, best alphabet first."""
    scored = [r for r in results if r.mi_tot is not None]
    skipped = [r for r in results if r.mi_tot is None]
    scored.sort(key=lambda r: r.mi_tot if r.mi_tot is not None else 0.0, reverse=True)

    header = (
        f"{'run':26s} {'quant':6s} {'K':>3s} {'zd':>3s} {'val_loss':>9s} {'ep':>3s} "
        f"{'mi':>7s} {'mi_tot':>7s} {'chain':>7s} {'chain%':>7s} {'cap%':>6s} "
        f"{'eff_K':>6s} {'top%':>6s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in scored:
        capacity = (
            100 * r.mi_tot / math.log2(r.n_states)
            if r.mi_tot is not None and r.n_states > 1
            else float("nan")
        )
        effective = _effective_states(r)
        chain = _chain_autocorrelation(r)
        chain_share = _chain_fraction(r)
        print(
            f"{r.name[:26]:26s} {r.quantizer[:6]:6s} {r.n_states:3d} {r.z_dim:3d} "
            f"{r.best_val_loss if r.best_val_loss is not None else float('nan'):9.5f} "
            f"{r.best_epoch if r.best_epoch is not None else -1:3d} "
            f"{r.mi if r.mi is not None else float('nan'):7.4f} "
            f"{r.mi_tot if r.mi_tot is not None else float('nan'):7.4f} "
            f"{chain if chain is not None else float('nan'):7.4f} "
            f"{100 * chain_share if chain_share is not None else float('nan'):6.1f}% "
            f"{capacity:6.1f} "
            f"{effective if effective is not None else float('nan'):6.2f} "
            f"{100 * r.top_state_share if r.top_state_share is not None else float('nan'):6.1f}"
        )

    for r in skipped:
        print(f"{r.name[:26]:26s} {r.quantizer[:6]:6s}  --  skipped: {r.skipped}")

    print(
        "\nmi/mi_tot/chain in bits. cap% = mi_tot as a share of log2(K), the most any "
        "K-state\nalphabet could carry. eff_K = 2**entropy, the usage-weighted state count "
        "(all K states\nmay still appear; this measures imbalance, not dead states). "
        "top% = share of the single\nmost-used state. chain = the lagged term mi_tot "
        "subtracts: local smoothness, not alignment\nsignal; chain% is its share of raw mi, "
        "the tell for an arm that raises mi only by emitting\nlonger correlated state runs."
    )


def _average_ranks(values: list[float]) -> np.ndarray:
    """Rank values ascending, averaging any ties (the Spearman convention)."""
    array = np.asarray(values, dtype=np.float64)
    order = array.argsort()
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    # Average the ranks within each group of equal values so ties do not bias the result.
    for value in np.unique(array):
        tied = array == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _report_rank_agreement(results: list[RunResult]) -> None:
    """Test whether validation reconstruction loss predicts alphabet quality at all.

    Lower val_loss should mean higher mi_tot if reconstruction is a good proxy for the
    metric the project reports. If the correlation is weak or positive, then tuning against
    val_loss is not tuning the deliverable, and every reconstruction-based comparison in
    this project needs re-reading.
    """
    pairs = [
        (r.best_val_loss, r.mi_tot)
        for r in results
        if r.best_val_loss is not None and r.mi_tot is not None
    ]
    if len(pairs) < 3:
        print(f"\nrank agreement: need >=3 scored runs to test, have {len(pairs)}.")
        return

    losses = [pair[0] for pair in pairs]
    mi_tots = [pair[1] for pair in pairs]
    pearson = float(np.corrcoef(losses, mi_tots)[0, 1])
    spearman = float(np.corrcoef(_average_ranks(losses), _average_ranks(mi_tots))[0, 1])

    print(f"\nval_loss vs mi_tot over {len(pairs)} runs:")
    print(f"  spearman {spearman:+.3f}   pearson {pearson:+.3f}")
    print(
        "  Strongly negative is what you want -- lower reconstruction loss buying a better\n"
        "  alphabet. Near zero or positive means val_loss is not a proxy for the deliverable,\n"
        "  and arms should be selected on mi_tot instead. With this few runs treat it as a\n"
        "  smell test, not a measurement."
    )


def main() -> None:
    """Evaluate every run under --runs-root and print the compiled comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/ablation", help="Directory of run dirs.")
    parser.add_argument(
        "--pdb-dir",
        default="data/external/foldseek_scop40/pdb_by_sid",
        help="Directory of structures to encode.",
    )
    parser.add_argument(
        "--pairfile",
        default="data/derived/pairfiles/tmaln-06.val.out",
        help="Validation alignment pairfile.",
    )
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        default=None,
        help="Virtual center override; otherwise recovered from each run's provenance.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate runs that already have a report."
    )
    parser.add_argument("--csv", default=None, help="Optional path to write the table as CSV.")
    args = parser.parse_args()

    runs = _discover_runs(Path(args.runs_root))
    if not runs:
        print(f"No runs with a config.json under {args.runs_root}.")
        return
    print(f"Found {len(runs)} run(s) under {args.runs_root}.\n")

    results = [
        _collect(run_dir, args.pdb_dir, args.pairfile, args.virt, args.force) for run_dir in runs
    ]

    _print_table(results)
    _report_rank_agreement(results)

    if args.csv:
        fieldnames = list(vars(results[0]).keys())
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(vars(result))
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
